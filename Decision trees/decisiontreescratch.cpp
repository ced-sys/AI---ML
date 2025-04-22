#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>

using namespace std;

//Represents a rock sample: [silica content (%), grain size(mm), label (0=sedimentary, 1=igneous)]
using Sample = vector<double>;

//Node in the decision tree
struct Node {
	int feature_idx = -1; //Feature to split on (0=silica, 1=grain size)
	double threshold = 0.0; //Threshold for the split
	int label = -1; //Predicted label if leaf node (-1 if not a leaf)
	Node* left = nullptr; //Left child
	Node* right = nullptr; //Right child
};

//Calculate Gini impurity for a set of samples
double calculate_gini(const vector<Sample>& samples) {
	if (samples.empty()) return 0.0;
	int count[2] = { 0, 0 }; //Count of sedimentary (0) and igneous (1) 
	for (const auto& sample : samples) {
		count[int(sample[2])]++; //label is at index 2
	}
	double total = samples.size();
	double gini = 1.0;
	for (int i = 0; i < 2; i++) {
		double p = count[i] / total;
		gini -= p * p;
	}
	return gini;
}

//Find the best split for a node
void find_best_split(const vector<Sample>& samples, int& best_feature, double& best_threshold, double& best_gini) {
	best_gini = numeric_limits<double>::max();
	best_feature = -1;
	best_threshold = 0.0;

	//Try each feature (0=silica, 1=grain size)
	for (int feature = 0; feature < 2; feature++) {
		//Get unique values for the feature to try as thresholds
		vector<double> thresholds;
		for (const auto& sample : samples) {
			thresholds.push_back(sample[feature]);
		}
		sort(thresholds.begin(), thresholds.end());
		thresholds.erase(unique(thresholds.begin(), thresholds.end()), thresholds.end());

		//Try each threshold
		for (double thresh : thresholds) {
			vector<Sample> left, right;
			for (const auto& sample : samples) {
				if (sample[feature] <= thresh) {
					left.push_back(sample);
				}
				else {
					right.push_back(sample);
				}
			}
			if (left.empty() || right.empty()) continue;

			//Calculate weighted Gini
			double gini_left = calculate_gini(left);
			double gini_right = calculate_gini(right);
			double weighted_gini = (left.size() * gini_left + right.size() * gini_right) / samples.size();

			if (weighted_gini < best_gini) {
				best_gini = weighted_gini;
				best_feature = feature;
				best_threshold = thresh;
			}
		}
	}
}
//Build the Decision Tree recursively
Node* build_tree(const vector<Sample>& samples, int depth, int max_depth) {
	Node* node = new Node();

	//Stopping criteria
	double current_gini = calculate_gini(samples);
	if (depth >= max_depth || samples.size() < 2 || current_gini == 0.0) {
		//Make leaf node: predict majority class
		int count[2] = { 0, 0 };
		for (const auto& sample : samples) {
			count[int(sample[2])]++;
		}
		node->label = (count[1] >= count[0]) ? 1 : 0;
		return node;
	}

	//Find best split
	int best_feature;
	double best_threshold, best_gini;
	find_best_split(samples, best_feature, best_threshold, best_gini);

	if (best_feature == -1) {
		int count[2] = { 0, 0 };
		for (const auto& sample : samples) {
			count[int(sample[2])]++;
		}
		node->label = (count[1] >= count[0]) ? 1 : 0;
		return node;
	}

	//Split samples
	vector<Sample> left_samples, right_samples;
	for (const auto& sample : samples) {
		if (sample[best_feature] <= best_threshold) {
			left_samples.push_back(sample);
		}
		else {
			right_samples.push_back(sample);
		}
	}

	//Create node and recurse
	node->feature_idx = best_feature;
	node->threshold = best_threshold;
	node->left = build_tree(left_samples, depth + 1, max_depth);
	node->right = build_tree(right_samples, depth + 1, max_depth);
	return node;
}

//Predict for a single sample
int predict(Node* node, const Sample& sample) {
	if (node->label != -1) {
		return node->label;
	}
	if (sample[node->feature_idx] <= node->threshold) {
		return predict(node->left, sample);
	}
	return predict(node->right, sample);
}

//Clean up memory
void delete_tree(Node* node) {
	if (!node) return;
	delete_tree(node->left);
	delete_tree(node->right);
	delete node;
}

int main() {
	//Simulated geological dataset: [silica (%), grain size(mm), label]
	vector<Sample> dataset = {
		{70.0, 1.0, 1}, //Igneous (high silica, medium grain)
		{55.0, 0.5, 1}, //Igneous
		{65.0, 2.0, 1}, //Igneous
		{30.0, 0.1, 0}, //Sedimentary (low silica, fine grain)
		{40.0, 1.5, 0}, //Sedimentary
		{50.0, 0.05, 0} //Sedimentary
	};

	//Build tree (max depth=2 for simplicity)
	Node* root = build_tree(dataset, 0, 2);

	//Test predictions
	cout << "Testing rock samples:\n";
	vector<Sample> test_samples = {
		{60.0, 1.2}, //likely igneous
		{35.0, 0.2} //Likely sedimentary
	};
	for (const auto& sample : test_samples) {
		int pred = predict(root, sample);
		cout << "Silica: " << sample[0] << "%, Grain Size:" << sample[1] << "mm->"
			<< (pred ? "Igneous" : "Sedimentary") << endl;
	}
	//Clean up
	delete_tree(root);
	return 0;
}