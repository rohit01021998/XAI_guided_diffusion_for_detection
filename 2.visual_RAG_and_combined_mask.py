import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import ollama
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import ssl

# Config for text-based embedding
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# Load pre-trained CNN model for feature extraction with error handling
try:
    cnn_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    cnn_model.eval()
    # Remove the final classification layer to get features
    cnn_feature_extractor = torch.nn.Sequential(*list(cnn_model.children())[:-1])
    CNN_AVAILABLE = True
    print("✓ ResNet50 model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load ResNet50 model: {e}")
    print("Falling back to basic visual features...")
    CNN_AVAILABLE = False
    cnn_feature_extractor = None

# Image preprocessing for CNN
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def cnn_features_to_text_description(cnn_features):
    """Convert CNN features to text description for embedding"""
    try:
        # CNN features shape: (2048,) for ResNet50
        features = cnn_features.flatten()
        
        # Analyze feature patterns to create meaningful text descriptions
        # High-level feature analysis
        feature_mean = np.mean(features)
        feature_std = np.std(features)
        # feature_max = np.max(features)
        # feature_min = np.min(features)
        
        # Pattern analysis
        high_activation_ratio = np.sum(features > feature_mean + feature_std) / len(features)
        # low_activation_ratio = np.sum(features < feature_mean - feature_std) / len(features)
        
        # Texture/complexity indicators from CNN features
        # Tells how much feature value varies
        activation_diversity = feature_std / (feature_mean + 1e-8)
        # Tells what fraction of features are nearly zero
        feature_sparsity = np.sum(features < 0.1) / len(features)
        
        # Determine visual characteristics from CNN feature patterns
        # If more features more than 15% are highly activated, it is complex and
        if high_activation_ratio > 0.15:
            complexity = "complex detailed"
        elif high_activation_ratio > 0.08:
            complexity = "moderate detailed"
        else:
            complexity = "simple"
            
        if activation_diversity > 1.5:
            texture = "rich textured"
        elif activation_diversity > 0.8:
            texture = "moderate textured"
        else:
            texture = "smooth"
            
        if feature_sparsity > 0.7:
            contrast = "high contrast"
        elif feature_sparsity > 0.4:
            contrast = "moderate contrast"
        else:
            contrast = "low contrast"
            
        # Analyze different feature regions for semantic understanding
        # Early features (edges, basic patterns)
        early_features = features[:512]
        early_activation = np.mean(early_features)
        
        # Mid-level features (shapes, parts)
        mid_features = features[512:1536]
        mid_activation = np.mean(mid_features)
        
        # High-level features (objects, semantics)
        high_features = features[1536:]
        high_activation = np.mean(high_features)
        
        # Determine dominant feature type
        activations = [early_activation, mid_activation, high_activation]
        max_idx = np.argmax(activations)
        
        if max_idx == 0:
            semantic_desc = "edge-dominated geometric"
        elif max_idx == 1:
            semantic_desc = "shape-focused structural"
        else:
            semantic_desc = "object-level semantic"
            
        # Create comprehensive text description from CNN features
        description = f"car image with {complexity} features, {texture} patterns, {contrast} lighting, {semantic_desc} characteristics"
        
        return description
        
    except Exception as e:
        print(f"Error converting CNN features to text: {e}")
        return None

def image_to_text_description(image_path):
    """Convert image to text description using CNN features"""
    try:
        # Check if CNN is available
        if not CNN_AVAILABLE or cnn_feature_extractor is None:
            raise RuntimeError("CNN model not available")
            
        # Load and preprocess image for CNN
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert BGR to RGB for PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        # Preprocess for CNN
        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
        
        # Extract CNN features
        with torch.no_grad():
            cnn_features = cnn_feature_extractor(input_batch)
            cnn_features = cnn_features.squeeze().numpy()  # Remove batch dim and convert to numpy
        
        # Convert CNN features to text description
        description = cnn_features_to_text_description(cnn_features)
        return description
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def get_image_embedding(image_path, target_size=(224, 224)):
    """Create text-based embedding using ollama like hybrid RAG"""
    try:
        # Convert image to text description
        description = image_to_text_description(image_path)
        if description is None:
            return None
        
        # Get text embedding using ollama
        try:
            embedding = ollama.embed(model=EMBEDDING_MODEL, input=description)['embeddings'][0]
            return np.array(embedding, dtype=np.float32)
        except Exception as ollama_error:
            raise RuntimeError(f"Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download. Error: {ollama_error}")
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def extract_red_importance(image):
    """Extract red importance regions from Grad-CAM saliency maps (RED=important areas)"""
    if len(image.shape) == 3:
        # OpenCV uses BGR format: [0]=Blue, [1]=Green, [2]=Red
        blue_channel = image[:, :, 0].astype(np.float32)
        green_channel = image[:, :, 1].astype(np.float32)
        red_channel = image[:, :, 2].astype(np.float32)
        
        # Focus on pure red areas (high red, low blue/green)
        # This identifies the important regions from Grad-CAM output
        red_dominance = red_channel - np.maximum(blue_channel, green_channel)
        red_dominance = np.clip(red_dominance, 0, None)
        
        # Also consider overall red intensity
        red_intensity = red_channel.copy()
        
        # Combine both measures: areas that are both red and dominant
        importance = 0.7 * red_intensity + 0.3 * red_dominance
        
        if importance.max() > 0:
            importance = (importance / importance.max() * 255).astype(np.uint8)
        else:
            importance = np.zeros_like(importance, dtype=np.uint8)
    else:
        # For grayscale images, use the intensity directly
        importance = image.copy()
        if importance.max() > 0:
            importance = (importance / importance.max() * 255).astype(np.uint8)
    
    return importance

def combine_saliency_maps(saliency_map_paths, weights=None):
    """Combine multiple saliency maps focusing on red important areas"""
    if not saliency_map_paths or all(path is None for path in saliency_map_paths):
        return None
    
    valid_paths = [path for path in saliency_map_paths if path is not None and os.path.exists(path)]
    if not valid_paths:
        return None
    
    combined_importance = None
    valid_count = 0
    
    for i, path in enumerate(valid_paths):
        try:
            saliency_img = cv2.imread(path)
            if saliency_img is None:
                continue
            
            # Extract red importance regions
            importance_mask = extract_red_importance(saliency_img)
            
            if combined_importance is None:
                combined_importance = importance_mask.astype(np.float32)
            else:
                # Resize to match if needed
                if importance_mask.shape != combined_importance.shape:
                    importance_mask = cv2.resize(importance_mask, (combined_importance.shape[1], combined_importance.shape[0]))
                
                # Use weighted average to combine maps
                weight = weights[i] if weights and i < len(weights) else 1.0
                combined_importance = (combined_importance * valid_count + importance_mask.astype(np.float32) * weight) / (valid_count + weight)
            
            valid_count += 1
                
        except Exception as e:
            print(f"Error processing saliency map {path}: {e}")
            continue
    
    if combined_importance is None:
        return None
    
    # Convert back to uint8 and post-process
    final_mask = combined_importance.astype(np.uint8)
    
    if np.any(final_mask > 0):
        # Apply threshold to focus on most important areas
        threshold = np.percentile(final_mask[final_mask > 0], 60) if np.any(final_mask > 0) else 128
        final_mask = np.where(final_mask >= threshold, final_mask, 0)
        
        # Light smoothing to reduce noise while preserving important regions
        if np.any(final_mask > 0):
            final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
            
            # Enhance contrast of important regions
            if final_mask.max() > final_mask.min():
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                final_mask = clahe.apply(final_mask)
    
    return final_mask

def debug_saliency_combination(saliency_map_paths, combined_mask, output_path=None):
    """Debug function to visualize original saliency maps and combined output"""
    valid_paths = [path for path in saliency_map_paths if path is not None and os.path.exists(path)]
    if not valid_paths or combined_mask is None:
        print("No valid saliency maps or combined mask for debugging")
        return
    
    # Create visualization
    num_maps = len(valid_paths)
    fig, axes = plt.subplots(2, max(3, num_maps), figsize=(4*max(3, num_maps), 8))
    
    # Show original saliency maps
    for i, path in enumerate(valid_paths[:5]):  # Show max 5 maps
        saliency_img = cv2.imread(path)
        if saliency_img is not None:
            saliency_rgb = cv2.cvtColor(saliency_img, cv2.COLOR_BGR2RGB)
            if i < axes.shape[1]:
                axes[0, i].imshow(saliency_rgb)
                axes[0, i].set_title(f'Original Map {i+1}')
                axes[0, i].axis('off')
    
    # Hide unused subplot in first row
    for i in range(len(valid_paths), axes.shape[1]):
        axes[0, i].axis('off')
    
    # Show red importance extraction for first map
    if valid_paths:
        first_map = cv2.imread(valid_paths[0])
        if first_map is not None:
            red_importance = extract_red_importance(first_map)
            axes[1, 0].imshow(red_importance, cmap='hot')
            axes[1, 0].set_title('Red Importance (1st map)')
            axes[1, 0].axis('off')
    
    # Show combined result
    axes[1, 1].imshow(combined_mask, cmap='hot')
    axes[1, 1].set_title('Combined Red Areas')
    axes[1, 1].axis('off')
    
    # Show enhanced version
    if combined_mask.max() > 0:
        enhanced = cv2.applyColorMap(combined_mask, cv2.COLORMAP_JET)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        axes[1, 2].imshow(enhanced_rgb)
        axes[1, 2].set_title('Enhanced Visualization')
        axes[1, 2].axis('off')
    
    # Hide unused subplots in second row
    for i in range(3, axes.shape[1]):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save debug visualization
    if output_path:
        debug_path = output_path.replace('.png', '_debug.png')
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        print(f"Debug visualization saved to: {debug_path}")
    else:
        plt.show()
    
    plt.close()

def retrieve_and_combine_saliency(query_image_path, category, vrag_instance, top_k=5, output_path=None, debug=False):
    """Use Visual RAG to retrieve similar images and combine their saliency maps"""
    saliency_paths = vrag_instance.search_similar_images(query_image_path, category, top_k)
    
    if not saliency_paths:
        print(f"No saliency maps found for category: {category}")
        return None
    
    print(f"Retrieved {len([p for p in saliency_paths if p is not None])} saliency maps")
    
    # Combine saliency maps using final script logic
    combined_mask = combine_saliency_maps(saliency_paths)
    
    if combined_mask is None:
        print("Failed to combine saliency maps")
        return None
    
    # Debug visualization if requested
    if debug:
        debug_saliency_combination(saliency_paths, combined_mask, output_path)
    
    # Save combined mask if output path provided
    if output_path:
        cv2.imwrite(output_path, combined_mask)
        print(f"Combined red importance mask saved to: {output_path}")
    
    return combined_mask

def retrieve_saliency_and_original(query_image_path, category, vrag_instance, top_k=5, output_path=None, debug=False, return_multiple=False):
    """
    Use Visual RAG to retrieve similar images, combine their saliency maps, and return the top original image.

    Args:
        query_image_path: Path to the query image.
        category: Category to search in.
        vrag_instance: Instance of VisualRAG.
        top_k: Number of results to retrieve.
        output_path: Optional path to save the combined mask.
        debug: Whether to enable debug visualization.
        return_multiple: If True, returns a list of (combined_mask_path, top_original_image_path) for top_k results.
                         Otherwise, returns only the best (combined_mask, top_original_image_path).

    Returns:
        If return_multiple is True: List of tuples (combined_mask_path, top_original_image_path).
        If return_multiple is False: Tuple of (combined_mask, top_original_image_path).
    """
    saliency_paths, original_paths = vrag_instance.search_similar_images_with_originals(query_image_path, category, top_k)

    if not saliency_paths or not original_paths:
        if return_multiple:
            return []
        else:
            raise ValueError(f"No similar images found for category: {category}")

    print(f"Retrieved {len([p for p in saliency_paths if p is not None])} saliency maps and original images")

    if return_multiple:
        results = []
        for i in range(top_k):
            if i < len(saliency_paths) and i < len(original_paths):
                current_saliency_map_path = saliency_paths[i]
                current_original_path = original_paths[i]

                # Combine only the i-th saliency map to get a mask for that specific option
                # Note: combine_saliency_maps expects a list of paths
                combined_mask_for_option = combine_saliency_maps([current_saliency_map_path])

                if combined_mask_for_option is not None:
                    # Save the individual combined mask to a temporary file for passing back path
                    temp_dir = "/tmp/rag_masks" # Using a /tmp directory for temporary masks
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_mask_path = os.path.join(temp_dir, f"combined_mask_{os.path.basename(query_image_path)}_{i}.png")
                    cv2.imwrite(temp_mask_path, combined_mask_for_option)

                    results.append((temp_mask_path, current_original_path))
        return results
    else:
        # Original logic for single best
        combined_mask = combine_saliency_maps(saliency_paths)

        if combined_mask is None:
            raise ValueError("Failed to combine saliency maps")

        top_original_path = original_paths[0] # The most similar original image

        if not os.path.exists(top_original_path):
            raise ValueError(f"Top original image not found: {top_original_path}")

        if debug:
            debug_saliency_combination(saliency_paths, combined_mask, output_path)

        if output_path:
            cv2.imwrite(output_path, combined_mask)
            print(f"Combined red importance mask saved to: {output_path}")

        return combined_mask, top_original_path

class VisualRAG:
    def __init__(self, base_path="/Users/rohittiwari/Documents/XAI_guided_diffusion/classified_cars_output"):
        self.base_path = base_path
        self.vector_db = {}
        self.categories = [  "back" ,  "front-right" ,  "back-left" ,  "back-right" ,  "front-left" ,  "front"  ]
        
    def build_database(self):
        """Build vector database for all categories"""
        print("Building Visual RAG database...")
        
        for category in self.categories:
            print(f"Processing category: {category}")
            category_path = os.path.join(self.base_path, category)
            
            if not os.path.exists(category_path):
                continue
                
            cropped_path = os.path.join(category_path, "cropped_images")
            saliency_path = os.path.join(category_path, "saliency_maps")
            
            if not os.path.exists(cropped_path) or not os.path.exists(saliency_path):
                continue
            
            cropped_images = [f for f in os.listdir(cropped_path) if f.endswith('.jpg')]
            category_db = []
            
            for img_file in cropped_images:
                img_path = os.path.join(cropped_path, img_file)
                saliency_map_path = os.path.join(saliency_path, img_file)
                
                if not os.path.exists(saliency_map_path):
                    saliency_map_path = None
                
                embedding = get_image_embedding(img_path)
                if embedding is not None:
                    category_db.append((embedding, saliency_map_path, img_path))
            
            self.vector_db[category] = category_db
            print(f"Added {len(category_db)} images to {category}")
        
        print("Database built successfully!")
        self.print_database_stats()
    
    def print_database_stats(self):
        """Print database statistics"""
        total_images = sum(len(db) for db in self.vector_db.values())
        for category, db in self.vector_db.items():
            print(f"{category}: {len(db)} images")
        print(f"Total: {total_images} images")
    
    def save_database(self, path="visual_rag_db.pkl"):
        """Save the database to disk"""
        with open(os.path.join(self.base_path, path), 'wb') as f:
            pickle.dump(self.vector_db, f)
        print(f"Database saved")
    
    def load_database(self, path="visual_rag_db.pkl"):
        """Load the database from disk"""
        load_path = os.path.join(self.base_path, path)
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                self.vector_db = pickle.load(f)
            print("Database loaded")
            self.print_database_stats()
            return True
        return False
    
    def search_similar_images(self, query_image_path, category, top_k=5):
        """
        Search for similar images and return their saliency maps and original images
        
        Args:
            query_image_path: Path to the query image
            category: Category to search in
            top_k: Number of saliency maps to return
        
        Returns:
            List of saliency map paths
        """
        query_embedding = get_image_embedding(query_image_path)
        if query_embedding is None:
            return []
        
        if category not in self.vector_db:
            return []
        
        results = []
        for embedding, saliency_path, original_path in self.vector_db[category]:
            similarity = cosine_similarity(query_embedding, embedding)
            results.append((similarity, saliency_path, original_path))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [saliency_path for _, saliency_path, _ in results[:top_k]]
    
    def search_similar_images_with_originals(self, query_image_path, category, top_k=5):
        """
        Search for similar images and return both saliency maps and original images
        
        Args:
            query_image_path: Path to the query image
            category: Category to search in
            top_k: Number of results to return
        
        Returns:
            Tuple of (saliency_paths, original_paths)
        """
        query_embedding = get_image_embedding(query_image_path)
        if query_embedding is None:
            return [], []
        
        if category not in self.vector_db:
            return [], []
        
        results = []
        for embedding, saliency_path, original_path in self.vector_db[category]:
            similarity = cosine_similarity(query_embedding, embedding)
            results.append((similarity, saliency_path, original_path))
        
        results.sort(key=lambda x: x[0], reverse=True)
        saliency_paths = [saliency_path for _, saliency_path, _ in results[:top_k]]
        original_paths = [original_path for _, _, original_path in results[:top_k]]
        
        return saliency_paths, original_paths
    
def demonstrate_visual_rag(debug=False):
    """Demonstrate the Visual RAG system with saliency map combination"""
    vrag = VisualRAG()
    
    database_loaded = vrag.load_database()
    
    if database_loaded:
        # Find a test image for compatibility check
        test_image_path = None
        for category in vrag.categories:
            category_path = os.path.join(vrag.base_path, category, "cropped_images")
            if os.path.exists(category_path):
                demo_files = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
                if demo_files:
                    test_image_path = os.path.join(category_path, demo_files[0])
                    break
        
        if test_image_path and not check_database_compatibility(vrag, test_image_path):
            print("🔄 Rebuilding database with compatible embeddings...")
            vrag.build_database()
            vrag.save_database()
    else:
        print("📊 Building new Visual RAG database...")
        vrag.build_database()
        vrag.save_database()
    
    # Example usage
    if 'front' in vrag.vector_db and vrag.vector_db['front']:
        demo_image_path = "/Users/rohittiwari/Documents/XAI_guided_diffusion/classified_cars_output/front/cropped_images"
        demo_files = os.listdir(demo_image_path)
        if demo_files:
            query_image = os.path.join(demo_image_path, demo_files[0])
            category = 'front'
            
            print(f"Query: {os.path.basename(query_image)} in {category}")
            
            # Use Visual RAG to retrieve and combine saliency maps
            output_path = f"combined_importance_map_{category}.png"
            combined_map = retrieve_and_combine_saliency(
                query_image, category, vrag, top_k=5, output_path=output_path, debug=debug
            )
            
            if combined_map is not None:
                print(f"Combined red importance areas successfully!")
                print(f"Higher intensity = More important red regions from saliency maps")

def process_query_image(image_path, category=None, debug=False):
    """
    Process a single query image using Visual RAG
    
    Args:
        image_path: Path to the query image
        category: Category to search in (if None, tries to auto-detect or use 'front')
        debug: Whether to show debug visualization
    """
    vrag = VisualRAG()
    
    # Load existing database and check compatibility
    database_loaded = vrag.load_database()
    
    if database_loaded:
        # Check if the database is compatible with current embedding method
        test_image_path = image_path  # Use the query image for compatibility test
        
        if not check_database_compatibility(vrag, test_image_path):
            print("🔄 Rebuilding database with compatible embeddings...")
            vrag.build_database()
            vrag.save_database()
    else:
        # No database found, build new one
        print("📊 Building new Visual RAG database...")
        vrag.build_database()
        vrag.save_database()
    
    # Use provided category or default to 'front'
    if category is None:
        category = 'front'  # Default category
    
    if category not in vrag.vector_db:
        print(f"Category '{category}' not found in database.")
        print(f"Available categories: {list(vrag.vector_db.keys())}")
        return None
    
    # Create output filename based on input image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"combined_importance_{base_name}_{category}.png"
    
    # Retrieve and combine saliency maps
    combined_map = retrieve_and_combine_saliency(
        image_path, category, vrag, top_k=3, output_path=output_path, debug=debug
    )
    
    return combined_map

def check_database_compatibility(vrag_instance, test_image_path):
    """
    Check if the existing database is compatible with current embedding method
    
    Args:
        vrag_instance: VisualRAG instance with loaded database
        test_image_path: Path to a test image to generate current embedding
    
    Returns:
        bool: True if compatible, False if needs rebuilding
    """
    try:
        # Generate a test embedding with current method
        test_embedding = get_image_embedding(test_image_path)
        if test_embedding is None:
            return False
        
        # Check if any category has embeddings with different shapes
        for category, db in vrag_instance.vector_db.items():
            if db:  # If category has embeddings
                stored_embedding = db[0][0]  # First embedding from this category
                if stored_embedding.shape != test_embedding.shape:
                    print(f"⚠️  Database incompatibility detected:")
                    print(f"   Stored embeddings: {stored_embedding.shape}")
                    print(f"   Current method: {test_embedding.shape}")
                    print(f"   Database needs to be rebuilt with new embedding method")
                    return False
        
        return True
        
    except Exception as e:
        print(f"Error checking database compatibility: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # Option for direct file input (coded in main, not CLI)
    # Change these paths to test specific files
    DIRECT_FILE_INPUT = "classified_cars_output/front-left/cropped_images/chrysler_sebring_2008_blue_02_frame_0342_det_0.jpg"  # Set to image path for direct testing
    DEBUG_MODE = True  # Set to True to show debug visualization
    
    if DIRECT_FILE_INPUT and os.path.exists(DIRECT_FILE_INPUT):
        # Direct file input mode
        print(f"Processing image: {DIRECT_FILE_INPUT}")
        combined_map = process_query_image(DIRECT_FILE_INPUT, category='front-left', debug=DEBUG_MODE)
        if combined_map is not None:
            print("Visual RAG processing completed successfully!")
    
    elif len(sys.argv) > 1:
        # Command line usage: python visual_RAG.py <image_path> [category] [--debug]
        image_path = sys.argv[1]
        category = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
        debug = '--debug' in sys.argv
        
        if os.path.exists(image_path):
            print(f"Processing image: {image_path}")
            combined_map = process_query_image(image_path, category, debug=debug)
            if combined_map is not None:
                print("Visual RAG processing completed successfully!")
        else:
            print(f"Image not found: {image_path}")
    else:
        # Demo mode
        print("Running in demo mode...")
        demonstrate_visual_rag(debug=DEBUG_MODE)
