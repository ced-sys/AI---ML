class Node:
    def __init__(self):
        self.feature_idx = -1  # Feature to split on (0=silica, 1=grain size)
        self.threshold = 0.0  # Threshold for the split
        self.label = -1  # Predicted label if leaf node (-1 if not a leaf)
        self.left = None  # Left child
        self.right = None  # Right child

def calculate_gini(samples):
    if not samples:
        return 0.0
    count = [0, 0]  # Count of sedimentary (0) and igneous (1)
    for sample in samples:
        count[int(sample[2])] += 1  # Label is at index 2
    total = len(samples)
    gini = 1.0
    for c in count:
        p = c/total
        gini -= p*p
    return gini

def find_best_split(samples):
    best_gini = float('inf')
    best_feature = -1
    best_threshold = 0.0

    # Try each feature (0=silica, 1=grain size)
    for feature in range(2):
        # Get unique values for the feature to try as thresholds
        thresholds = sorted(set(sample[feature] for sample in samples))

        # Try each threshold
        for thresh in thresholds:
            left = [s for s in samples if s[feature] <= thresh]
            right = [s for s in samples if s[feature] > thresh]
            if not left or not right:
                continue

            # Calculate weighted Gini
            gini_left = calculate_gini(left)
            gini_right = calculate_gini(right)
            weighted_gini = (len(left) * gini_left + len(right) * gini_right)

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = thresh

    return best_feature, best_threshold, best_gini

def build_tree(samples, depth, max_depth):
    node = Node()

    # Stopping criteria
    current_gini = calculate_gini(samples)
    if depth >= max_depth or len(samples) < 2 or current_gini == 0.0:
        # Make leaf node: predict majority class
        count = [0, 0]
        for sample in samples:
            count[int(sample[2])] += 1
        node.label = 1 if count[1] > count[0] else 0
        return node
    
    # Find the best split
    best_feature, best_threshold, _ = find_best_split(samples)

    # Split samples
    left_samples = [s for s in samples if s[best_feature] <= best_threshold]
    right_samples = [s for s in samples if s[best_feature] > best_threshold]

    # Create node and recurse
    node.feature_idx = best_feature
    node.threshold = best_threshold
    node.left = build_tree(left_samples, depth+1, max_depth)
    node.right = build_tree(right_samples, depth+1, max_depth)
    return node

def predict(node, sample):
    if node.label != -1:
        return node.label
    if sample[node.feature_idx] <= node.threshold:
        return predict(node.left, sample)
    return predict(node.right, sample)

def tree_to_dot(node, dot=None):
    if dot is None:
        from graphviz import Digraph
        dot = Digraph()
        dot.attr(rankdir='TB')
    
    # Create unique node IDs
    node_id = str(id(node))
    
    # Create node label
    if node.label != -1:
        label = f"Class: {'Igneous' if node.label == 1 else 'Sedimentary'}"
        dot.node(node_id, label, shape='box')
    else:
        feature_name = 'Silica' if node.feature_idx == 0 else 'Grain Size'
        label = f"{feature_name}\n≤ {node.threshold:.2f}"
        dot.node(node_id, label, shape='oval')
    
    # Create edges to children
    if node.left:
        dot.edge(node_id, str(id(node.left)))
        tree_to_dot(node.left, dot)
    if node.right:
        dot.edge(node_id, str(id(node.right)))
        tree_to_dot(node.right, dot)
    
    return dot

# main program
if __name__ == "__main__":
    # Simulated geological dataset: [silica,(%), grain size(mm), label]
    dataset = [
        [70.0, 1.0, 1],  # igneous(high silica, medium grained)
        [55.0, 0.5, 1],  # Igneous
        [65.0, 2.0, 1],  # igneous
        [30.0, 0.1, 0],  # Sedimentary (low silica, fine grain)
        [40.0, 1.5, 0],  # Sedmentary
        [50.0, 0.05, 0]  # Sedimentary
    ]

    # Build tree(max depth =2 for simplicity)
    root = build_tree(dataset, 0, 2)

    # Test predictions
    print("Testing rock samples")
    test_samples = [
        [60.0, 1.2],  # likely igneous
        [35.0, 0.2],  # Likely sedimentary
    ]

    for sample in test_samples:
        pred = predict(root, sample)
        print(f"Silica: {sample[0]}%, Grain size: {sample[1]}mm -> {'Igneous' if pred else 'Sedimentary'}")

    # Visualize the tree
    try:
        dot = tree_to_dot(root)
        dot.render("rock_decision_tree", format="png", cleanup=True)
        print("\nDecision tree visualization saved as 'rock_decision_tree.png'")
    except Exception as e:
        print("\nCouldn't create visualization. Make sure graphviz is installed:")
        print("pip install graphviz")
        print("Also ensure Graphviz is installed on your system")
