#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>

//Sigmoid for a single value
double sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}

//sigmoid for a vector values
std::vector<double> sigmoid(const std::vector<double>& x) {
	std::vector<double> result;
	result.reserve(x.size());
	for (double val : x) {
		result.push_back(sigmoid(val));
	}
	return result;
}

int main() {
	std::cout << "Enter numbers separated by space:";
	std::string line;
	std::stringstream ss(line);

	std::vector<double> input;
	double num;

	while (ss >> num) {
		input.push_back(num);
	}

	std::vector<double> output = sigmoid(input);

	std::cout << "Sigmoid values: [";
	for (size_t i = 0; i < output.size(); ++i) {
		std::cout << output[i];
		if (i < output.size() - 1) std::cout << ", ";
	}
	std::cout << "]" << std::endl;

	return 0;
}