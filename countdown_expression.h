#pragma once
#include <vector>
#include <string>

void print(const std::vector<int>& v);

class Expression {

private:
	std::vector<int> numbers;
	std::string operators;
	std::vector<int> tokens;
	std::vector<int> convert_symbols();
	std::vector<int> evaluate_rpn();
	bool check_rpn();

public:
	Expression(std::vector<int> numbers, std::string operators);
	void evaluate(std::vector<int> &counts);
};

class NumberSet {

private:
	static std::vector<std::string> symbols;

public:
	std::vector<int> numbers;
	std::vector<int> counts;
	static std::vector<int> global_counts;

	NumberSet(std::vector<int> numbers);
	void evaluate();
	static void generate_symbols();
	static std::vector<NumberSet> generate_numbers();
};

std::ostream& operator<<(std::ostream &strm, const NumberSet &n);