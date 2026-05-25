#!/usr/bin/env python3
import os
import json
from collections import defaultdict, Counter
from dataset.causal_vl import CausalVLDataset

def analyze_dataset(dataset_root: str):
    """Analyze the entire CausalVL dataset and provide comprehensive statistics."""
    
    # Load the dataset
    print("Loading dataset...")
    dataset = CausalVLDataset(dataset_root)
    
    # Initialize counters
    total_questions = 0
    total_videos = 0
    total_image_sets = 0
    unique_images = set()
    subcategory_stats = defaultdict(lambda: {
        'count': 0,
        'videos': 0,
        'image_sets': 0,
        'unique_images': set(),
        'total_images': 0
    })
    
    category_stats = defaultdict(lambda: {
        'count': 0,
        'videos': 0,
        'image_sets': 0,
        'unique_images': set(),
        'total_images': 0,
        'subcategories': set()
    })
    
    print(f"Processing {len(dataset.samples)} samples...")
    
    for sample in dataset.samples:
        # Count total questions
        total_questions += 1
        
        # Determine if this is a video or image set
        is_video = False
        is_image_set = False
        
        # Check if any media path contains "_frames" (indicating video extraction)
        for media_path in sample.media_paths:
            if "_frames" in media_path:
                is_video = True
                break
        
        if not is_video:
            is_image_set = True
        
        # Update counters
        if is_video:
            total_videos += 1
        else:
            total_image_sets += 1
        
        # Count unique images
        for media_path in sample.media_paths:
            unique_images.add(media_path)
        
        # Update subcategory statistics
        subcat_stats = subcategory_stats[sample.subcategory]
        subcat_stats['count'] += 1
        if is_video:
            subcat_stats['videos'] += 1
        else:
            subcat_stats['image_sets'] += 1
        
        subcat_stats['total_images'] += len(sample.media_paths)
        for media_path in sample.media_paths:
            subcat_stats['unique_images'].add(media_path)
        
        # Update category statistics
        cat_stats = category_stats[sample.category]
        cat_stats['count'] += 1
        cat_stats['subcategories'].add(sample.subcategory)
        if is_video:
            cat_stats['videos'] += 1
        else:
            cat_stats['image_sets'] += 1
        
        cat_stats['total_images'] += len(sample.media_paths)
        for media_path in sample.media_paths:
            cat_stats['unique_images'].add(media_path)
    
    # Convert sets to counts for final statistics
    for subcat_stats in subcategory_stats.values():
        subcat_stats['unique_image_count'] = len(subcat_stats['unique_images'])
        del subcat_stats['unique_images']  # Remove set to make JSON serializable
    
    for cat_stats in category_stats.values():
        cat_stats['unique_image_count'] = len(cat_stats['unique_images'])
        cat_stats['subcategory_count'] = len(cat_stats['subcategories'])
        del cat_stats['unique_images']  # Remove set to make JSON serializable
        del cat_stats['subcategories']  # Remove set to make JSON serializable
    
    # Compile final statistics
    statistics = {
        'overall': {
            'total_questions': total_questions,
            'total_videos': total_videos,
            'total_image_sets': total_image_sets,
            'unique_images': len(unique_images),
            'total_categories': len(category_stats),
            'total_subcategories': len(subcategory_stats)
        },
        'by_category': dict(category_stats),
        'by_subcategory': dict(subcategory_stats)
    }
    
    return statistics

def print_statistics(statistics):
    """Print statistics in a readable format."""
    
    print("\n" + "="*60)
    print("CAUSAL VL DATASET STATISTICS")
    print("="*60)
    
    # Overall statistics
    overall = statistics['overall']
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total Questions: {overall['total_questions']:,}")
    print(f"  Total Videos: {overall['total_videos']:,}")
    print(f"  Total Image Sets: {overall['total_image_sets']:,}")
    print(f"  Unique Images: {overall['unique_images']:,}")
    print(f"  Categories: {overall['total_categories']}")
    print(f"  Subcategories: {overall['total_subcategories']}")
    
    # Category statistics
    print(f"\nBY CATEGORY:")
    for category, stats in statistics['by_category'].items():
        print(f"  {category}:")
        print(f"    Questions: {stats['count']:,}")
        print(f"    Videos: {stats['videos']:,}")
        print(f"    Image Sets: {stats['image_sets']:,}")
        print(f"    Total Images: {stats['total_images']:,}")
        print(f"    Unique Images: {stats['unique_image_count']:,}")
        print(f"    Subcategories: {stats['subcategory_count']}")
    
    # Subcategory statistics
    print(f"\nBY SUBCATEGORY:")
    for subcategory, stats in statistics['by_subcategory'].items():
        print(f"  {subcategory}:")
        print(f"    Questions: {stats['count']:,}")
        print(f"    Videos: {stats['videos']:,}")
        print(f"    Image Sets: {stats['image_sets']:,}")
        print(f"    Total Images: {stats['total_images']:,}")
        print(f"    Unique Images: {stats['unique_image_count']:,}")

def save_statistics(statistics, output_file):
    """Save statistics to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"\nStatistics saved to: {output_file}")

def main():
    # Set the dataset root path
    dataset_root = "/home/tianyi/project/causalphy/dataset"
    
    # Check if dataset exists
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset root not found: {dataset_root}")
        return
    
    try:
        # Analyze the dataset
        statistics = analyze_dataset(dataset_root)
        
        # Print statistics
        print_statistics(statistics)
        
        # Save to file
        output_file = "/home/tianyi/project/causalphy/dataset_statistics.json"
        save_statistics(statistics, output_file)
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
