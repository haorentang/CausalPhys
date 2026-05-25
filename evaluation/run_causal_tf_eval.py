#!/usr/bin/env python3
import os
import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import urllib.request
import urllib.error

PROMPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../prompts/causal_tf_judge_prompt.txt"))
RESPONSES_PATH_FALLBACK = os.path.abspath(os.path.join(os.path.dirname(__file__), "../evaluation_results/gpt-4o/behaviour_selection/responses.json"))
OUT_DIR_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "./outputs"))
KEYS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../keys.txt"))

from tqdm import tqdm  # type: ignore

# Question ID prefixes
NODE_MENTION_PREFIX = "N:"  # N:<nodeId>
EDGE_CAUSAL_PREFIX = "E:"   # E:<childId<-parentId>
TEXT_DESC_PREFIX = "T:"     # T:<nodeId>


def load_prompt_text(prompt_path: str) -> str:
	with open(prompt_path, "r", encoding="utf-8") as f:
		return f.read()


def load_api_keys(keys_path: str) -> Dict[str, str]:
	"""Load API keys from keys.txt file"""
	keys = {}
	if os.path.exists(keys_path):
		with open(keys_path, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if "=" in line and not line.startswith("#"):
					key, value = line.split("=", 1)
					keys[key.strip()] = value.strip().strip('"')
	return keys


def build_parent_map(nodes: List[Dict[str, Any]], edges: List[Dict[str, str]]) -> Dict[str, List[str]]:
	parents: Dict[str, List[str]] = defaultdict(list)
	for edge in edges:
		parents[edge["to"].strip()].append(edge["from"].strip())
	return parents


def build_tf_questions_for_sample(sample: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, Any]]]:
	graph = sample["ground_truth_graph"]["graph"]
	nodes = graph["nodes"]
	edges = graph.get("edges", [])
	parent_map = build_parent_map(nodes, edges)

	node_by_id = {n["id"]: n for n in nodes}

	questions: List[Dict[str, str]] = []
	# meta holds mapping from qid -> meta info for scoring
	meta: Dict[str, Dict[str, Any]] = {}
	qid_counter = 0

	for node in nodes:
		node_id = node["id"]
		node_name = node.get("name", node_id)
		node_text = node.get("text")

		# existence question for every node
		qid = str(qid_counter)
		qtext = f"Does the rationale mention '{node_name}' (existence)?"
		questions.append({"id": qid, "text": qtext})
		meta[qid] = {"kind": "existence", "node_id": node_id}
		qid_counter += 1

		# parent_node questions for each parent
		for parent_id in parent_map.get(node_id, []):
			parent_name = node_by_id.get(parent_id, {}).get("name", parent_id)
			qid = str(qid_counter)
			qtext = f"Is the parent relation '{parent_name} -> {node_name}' correctly expressed?"
			questions.append({"id": qid, "text": qtext})
			meta[qid] = {"kind": "parent_node", "child_id": node_id, "parent_id": parent_id}
			qid_counter += 1

		# description correctness question for nodes with text
		if isinstance(node_text, str) and node_text.strip():
			qid = str(qid_counter)
			qtext = f"Is '{node_name}': '{node_text}' (or similar meaning) mentioned in the rationale?"
			questions.append({"id": qid, "text": qtext})
			meta[qid] = {"kind": "description", "node_id": node_id}
			qid_counter += 1

	return questions, meta


def _yaml_escape(s: str) -> str:
	return s.replace('"', '\\"')


def _dump_yaml(problem: str, rationale: str, questions: List[Dict[str, str]]) -> str:
	lines: List[str] = []
	lines.append("problem: |-")
	for line in (problem or "").splitlines():
		lines.append(f"  {line}")
	lines.append("rationale: |-")
	for line in (rationale or "").splitlines():
		lines.append(f"  {line}")
	lines.append("questions:")
	for q in questions:
		qid = str(q.get("id", ""))
		qtext = str(q.get("text", ""))
		lines.append("  - id: \"" + _yaml_escape(qid) + "\"")
		lines.append("    text: \"" + _yaml_escape(qtext) + "\"")
	return "\n".join(lines) + "\n"


def write_yaml_payload(out_path: str, problem: str, rationale: str, questions: List[Dict[str, str]]):
	yaml_text = _dump_yaml(problem, rationale, questions)
	with open(out_path, "w", encoding="utf-8") as f:
		f.write(yaml_text)


def try_load_answers_yaml(path: str) -> Dict[str, Any]:
	# Try YAML, fallback to JSON
	try:
		import yaml  # type: ignore
		with open(path, "r", encoding="utf-8") as f:
			return yaml.safe_load(f)
	except Exception:
		try:
			with open(path, "r", encoding="utf-8") as f:
				return json.load(f)
		except Exception:
			return {}


def parse_answers_fallback(text: str) -> Dict[str, Any]:
	# Very lightweight parser for our expected YAML answers format
	answers: List[Dict[str, Any]] = []
	current: Dict[str, Any] = {}
	for raw in text.splitlines():
		line = raw.strip()
		if line.startswith("- id:"):
			if current:
				answers.append(current)
			current = {}
			qid = line.split(":", 1)[1].strip().strip('"')
			current["id"] = qid
		elif line.startswith("answer:"):
			val = line.split(":", 1)[1].strip().lower()
			current["answer"] = True if val == "true" else False
	if current:
		answers.append(current)
	return {"answers": answers}


def score_answers(answers_yaml: Dict[str, Any], meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
	# Compute three metrics: existence, parent_node, description
	totals = {"existence": [0, 0], "parent_node": [0, 0], "description": [0, 0]}  # [correct, total]
	for item in answers_yaml.get("answers", []):
		qid = item.get("id")
		ans = item.get("answer")
		if qid not in meta:
			continue
		kind = meta[qid]["kind"]
		if kind not in totals:
			continue
		totals[kind][1] += 1
		if isinstance(ans, bool) and ans:
			totals[kind][0] += 1
	# For kinds with no applicable questions, return None so they are excluded from averages
	result: Dict[str, Any] = {}
	for k, (correct, total) in totals.items():
		if total == 0:
			result[k] = None
		else:
			result[k] = correct / total
	return result


def resolve_paths(model: str, subfolder: str, responses_arg: str, out_dir_arg: str) -> Tuple[str, str]:
	# Determine responses path
	if responses_arg:
		responses_path = responses_arg
	else:
		if model and subfolder:
			responses_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../evaluation_results/{model}/{subfolder}/responses.json"))
		elif model and not subfolder:
			# if only model provided, default to behaviour_selection like original
			responses_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../evaluation_results/{model}/behaviour_selection/responses.json"))
		else:
			responses_path = RESPONSES_PATH_FALLBACK

	# Output directory defaults to same directory as responses.json
	if out_dir_arg:
		out_dir = out_dir_arg
	else:
		out_dir = os.path.dirname(responses_path)

	return responses_path, out_dir


def find_all_model_subsets(results_root: str) -> List[Tuple[str, str, str, str]]:
	"""Scan evaluation_results tree to find all (model, subfolder) with responses.json.

	Returns list of tuples: (model, subfolder, responses_path, out_dir)
	"""
	combos: List[Tuple[str, str, str, str]] = []
	if not os.path.isdir(results_root):
		return combos
	for model in sorted(os.listdir(results_root)):
		model_dir = os.path.join(results_root, model)
		if not os.path.isdir(model_dir):
			continue
		for subfolder in sorted(os.listdir(model_dir)):
			sub_dir = os.path.join(model_dir, subfolder)
			if not os.path.isdir(sub_dir):
				continue
			responses_path = os.path.join(sub_dir, "responses.json")
			if os.path.exists(responses_path):
				combos.append((model, subfolder, responses_path, sub_dir))
	return combos


def process_subset(model: str, subfolder: str, responses_path: str, out_dir: str, args) -> None:
	"""Process a single (model, subfolder) subset, optionally performing auto-eval and scoring."""
	os.makedirs(out_dir, exist_ok=True)

	with open(responses_path, "r", encoding="utf-8") as f:
		samples = json.load(f)

	prompt_text = load_prompt_text(PROMPT_PATH)

	# Load API keys from keys.txt or environment
	api_keys = load_api_keys(KEYS_PATH)
	api_key = api_keys.get(args.api_key_env) or os.environ.get(args.api_key_env, "")

	# Prepare grouping by sub_category
	subcat_scores: Dict[str, List[Dict[str, float]]] = defaultdict(list)

	# Default directory to read pre-generated answers from (if any)
	default_answers_dir = os.path.join(out_dir, "judge")

	# Ensure directories used for writing exist
	instances_dir = os.path.join(out_dir, "instances")
	if args.auto_eval and api_key:
		os.makedirs(instances_dir, exist_ok=True)

	# Pre-compute questions/meta per sample
	prepared: List[Tuple[Dict[str, Any], List[Dict[str, str]], Dict[str, Dict[str, Any]]]] = []
	for sample in samples:
		# Skip samples without graph structure
		if "ground_truth_graph" not in sample or "graph" not in sample["ground_truth_graph"]:
			continue
		questions, meta = build_tf_questions_for_sample(sample)
		prepared.append((sample, questions, meta))

	def judge_one(sample: Dict[str, Any], questions: List[Dict[str, str]], meta: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
		"""Call the judge model for a sample, save per-question instance JSON, and return parsed answers."""
		sample_id = sample.get("sample_id", "unknown")
		problem = sample.get("ground_truth_graph", {}).get("question", "")
		rationale = sample.get("model_response_rationale", "")
		messages = build_messages(prompt_text, problem, rationale, questions)
		judge_model = args.judge_model or args.api_model
		if args.api_provider == "openrouter":
			text = call_openrouter_chat(api_key, judge_model, messages)
		else:
			text = call_openai_chat(args.api_base, api_key, judge_model, messages)
		answers_yaml_local: Dict[str, Any] = try_load_answers_yaml_from_text(text)
		if not answers_yaml_local:
			answers_yaml_local = parse_answers_fallback(text)
		# Build per-question instances and write JSON
		instances: List[Dict[str, Any]] = []
		answer_map: Dict[str, Any] = {}
		for item in answers_yaml_local.get("answers", []):
			qid = str(item.get("id"))
			answer_map[qid] = item.get("answer")
		for q in questions:
			qid = str(q.get("id", ""))
			qtext = str(q.get("text", ""))
			m = meta.get(qid, {})
			instances.append({
				"id": qid,
				"question": qtext,
				"answer": bool(answer_map.get(qid, False)) if isinstance(answer_map.get(qid, None), bool) else answer_map.get(qid, None),
				"kind": m.get("kind"),
				"node_id": m.get("node_id"),
				"parent_id": m.get("parent_id"),
				"child_id": m.get("child_id"),
			})
		out_path = os.path.join(instances_dir, f"{sample_id}.json")
		try:
			with open(out_path, "w", encoding="utf-8") as jf:
				json.dump({"sample_id": sample_id, "instances": instances}, jf, ensure_ascii=False, indent=2)
		except Exception as e:
			print(f"Failed writing instances for {sample_id}: {e}")
		return sample_id, answers_yaml_local

	# Map sample_id to parsed answers
	sample_id_to_answers: Dict[str, Dict[str, Any]] = {}

	if args.auto_eval and api_key:
		# Parallel judge calls with progress bar
		workers = max(1, int(args.workers)) if hasattr(args, "workers") else 1
		with ThreadPoolExecutor(max_workers=workers) as executor:
			futures = [executor.submit(judge_one, smpl, qst, meta) for (smpl, qst, meta) in prepared]
			for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Judging {model}/{subfolder}", unit="sample"):
				try:
					sample_id, ans = fut.result()
					sample_id_to_answers[sample_id] = ans
				except Exception as e:
					# Log and continue
					print(f"Worker error: {e}")

	# Scoring pass
	for (sample, questions, meta) in prepared:
		sample_id = sample.get("sample_id", "unknown")
		answers_yaml: Dict[str, Any] = {}
		if sample_id in sample_id_to_answers:
			answers_yaml = sample_id_to_answers[sample_id]
		else:
			# Try default 'judge' directory co-located with responses (instances preferred)
			ans_path = os.path.join(default_answers_dir, "instances", f"{sample_id}.json")
			if os.path.exists(ans_path):
				try:
					with open(ans_path, "r", encoding="utf-8") as jf:
						data = json.load(jf)
						answers_yaml = {"answers": [{"id": inst.get("id"), "answer": inst.get("answer")} for inst in data.get("instances", [])]}
				except Exception:
					answers_yaml = {}
			else:
				# Fallback to legacy yaml/json formats if present
				legacy = os.path.join(default_answers_dir, "yaml", f"{sample_id}.answers.yaml")
				if not os.path.exists(legacy):
					legacy = os.path.join(default_answers_dir, f"{sample_id}.answers.yaml")
				if not os.path.exists(legacy):
					legacy = os.path.join(default_answers_dir, f"{sample_id}.answers.json")
				if os.path.exists(legacy):
					answers_yaml = try_load_answers_yaml(legacy)

		if answers_yaml:
			scores = score_answers(answers_yaml, meta)
			subcat = sample.get("ground_truth_graph", {}).get("sub_category", "unknown")
			subcat_scores[subcat].append(scores)

	# Write judge.txt with averages for this subset
	all_scores: List[Dict[str, Any]] = []
	for _subcat, score_list in subcat_scores.items():
		all_scores.extend(score_list)
	if all_scores:
		avg = {"existence": 0.0, "parent_node": 0.0, "description": 0.0}
		counts = {"existence": 0, "parent_node": 0, "description": 0}
		for s in all_scores:
			for k in avg.keys():
				val = s.get(k)
				if isinstance(val, (int, float)):
					avg[k] += float(val)
					counts[k] += 1
		# Divide by non-null counts; if zero, leave at 0.0
		for k in avg.keys():
			c = counts[k]
			avg[k] = (avg[k] / c) if c > 0 else 0.0
		judge_path = os.path.join(out_dir, "judge.txt")
		with open(judge_path, "w", encoding="utf-8") as jf:
			jf.write(
				"existence: {:.4f}\nparent_node: {:.4f}\ndescription: {:.4f}\n".format(
					avg["existence"], avg["parent_node"], avg["description"]
				)
			)


def build_messages(prompt_text: str, problem: str, rationale: str, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
	# Single message content combining prompt, problem, rationale, and questions as YAML block
	payload = _dump_yaml(problem, rationale, questions)
	content = prompt_text + "\n\n" + payload
	return [{"role": "user", "content": content}]


def call_openai_chat(api_base: str, api_key: str, model: str, messages: List[Dict[str, str]], max_retries: int = 3, timeout: int = 60) -> str:
	url = api_base.rstrip("/") + "/v1/chat/completions"
	data = {
		"model": model,
		"messages": messages,
		"temperature": 0,
	}
	body = json.dumps(data).encode("utf-8")
	req = urllib.request.Request(url, data=body, headers={
		"Content-Type": "application/json",
		"Accept": "application/json",
		"Authorization": f"Bearer {api_key}",
	})
	for attempt in range(max_retries):
		try:
			with urllib.request.urlopen(req, timeout=timeout) as resp:
				resp_data = json.loads(resp.read().decode("utf-8"))
				return resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
		except urllib.error.HTTPError as e:
			try:
				err_body = e.read().decode("utf-8")
				print(f"OpenAI HTTPError {e.code}: {err_body}")
			except Exception:
				print(f"OpenAI HTTPError {e.code}: (no body)")
			time.sleep(1.5 * (attempt + 1))
			continue
		except urllib.error.URLError as e:
			print(f"OpenAI URLError: {e}")
			time.sleep(1.5 * (attempt + 1))
			continue
		except Exception as e:
			print(f"OpenAI Exception: {e}")
			break
	return ""


def call_openrouter_chat(api_key: str, model: str, messages: List[Dict[str, str]], max_retries: int = 3, timeout: int = 60) -> str:
	"""Call OpenRouter API"""
	url = "https://openrouter.ai/api/v1/chat/completions"
	data = {
		"model": model,
		"messages": messages,
		"temperature": 0,
	}
	body = json.dumps(data).encode("utf-8")
	req = urllib.request.Request(url, data=body, headers={
		"Content-Type": "application/json",
		"Accept": "application/json",
		"Authorization": f"Bearer {api_key}",
		"HTTP-Referer": "https://github.com/your-repo",
		"X-Title": "Causal TF Evaluator",
	})
	for attempt in range(max_retries):
		try:
			with urllib.request.urlopen(req, timeout=timeout) as resp:
				resp_data = json.loads(resp.read().decode("utf-8"))
				if "error" in resp_data:
					print(f"OpenRouter API Error: {resp_data['error']}")
					return ""
				return resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
		except urllib.error.HTTPError as e:
			try:
				err_body = e.read().decode("utf-8")
				print(f"OpenRouter HTTPError {e.code}: {err_body}")
			except Exception:
				print(f"OpenRouter HTTPError {e.code}: (no body)")
			time.sleep(1.5 * (attempt + 1))
			continue
		except urllib.error.URLError as e:
			print(f"OpenRouter URLError on attempt {attempt + 1}: {e}")
			time.sleep(1.5 * (attempt + 1))
			continue
		except Exception as e:
			print(f"OpenRouter Exception on attempt {attempt + 1}: {e}")
			break
	return ""


def main():
	parser = argparse.ArgumentParser(description="Causal TF evaluator - question generator, API caller, and scorer")
	parser.add_argument("--model", default="", help="Model name (e.g., gpt-4o) to locate evaluation_results/<model>/...")
	parser.add_argument("--subfolder", default="", help="Subfolder under the model (e.g., behaviour_selection) to process only that folder")
	parser.add_argument("--responses", default="", help="Explicit path to responses.json (overrides --model/--subfolder)")
	parser.add_argument("--out_dir", default="", help="Output directory (default: same as responses.json)")
	parser.add_argument("--write_prompts", action="store_true", help="Also write prompt text (deprecated - no longer generates YAML inputs)")
	parser.add_argument("--auto_eval", action="store_true", help="Call VLM once per sample to evaluate rationales and write *.answers.yaml")
	parser.add_argument("--api_provider", default="openai", help="API provider type: openai, openrouter")
	parser.add_argument("--api_base", default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com"), help="API base URL (for openai provider)")
	parser.add_argument("--api_model", default=os.environ.get("OPENAI_API_MODEL", "gpt-4o-mini"), help="API model ID")
	parser.add_argument("--judge_model", default=None, help="Judge model ID (overrides --api_model for judging)")
	parser.add_argument("--api_key_env", default="OPENAI_API_KEY", help="Environment variable name holding API key (or use keys.txt)")
	parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for API calls during auto-eval")
	args = parser.parse_args()

	# Determine whether to iterate over all models/subfolders
	results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../evaluation_results"))
	model = args.model
	subfolder = args.subfolder

	if (model and model.upper() == "ALL") or (subfolder and subfolder.upper() == "ALL"):
		# Discover all combos
		all_combos = find_all_model_subsets(results_root)
		# Optional filtering
		if model and model.upper() != "ALL":
			all_combos = [c for c in all_combos if c[0] == model]
		if subfolder and subfolder.upper() != "ALL":
			all_combos = [c for c in all_combos if c[1] == subfolder]
		for m, s, resp_path, out_dir_found in tqdm(all_combos, total=len(all_combos), desc="Subsets", unit="subset"):
			judge_txt = os.path.join(out_dir_found, "judge.txt")
			if os.path.exists(judge_txt):
				print(f"Skipping {m}/{s} (judge.txt exists)")
				continue
			print(f"Processing {m}/{s}")
			process_subset(m, s, resp_path, out_dir_found, args)
		return

	# Single subset path
	responses_path, out_dir = resolve_paths(model, subfolder, args.responses, args.out_dir)
	process_subset(model, subfolder, responses_path, out_dir, args)


def try_load_answers_yaml_from_text(text: str) -> Dict[str, Any]:
	# attempt YAML parse from text via dynamic import
	try:
		import yaml  # type: ignore
		return yaml.safe_load(text) or {}
	except Exception:
		return {}


if __name__ == "__main__":
	main()
