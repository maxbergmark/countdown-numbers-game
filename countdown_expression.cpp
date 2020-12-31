#include <algorithm>
#include <set>
#include <iostream>
#include <cstdint>

#include "countdown_expression.h"

#define NUMBER_MAX_SIZE 1000

void print(const std::vector<int>& v)
{
	for (int e : v) {
		std::cout << " " << e;
	}
	std::cout << std::endl;
}

/**
 *	The expression class represents a set of 6 board numbers, 
 *	along with 5 operators. The Expression class is responsible
 *	for generating all permutations of these 11 tokens, and 
 * 	for checking and evaluating the RPN.
 */

Expression::Expression(std::vector<int> numbers, std::string operators) {
	this->numbers = numbers;
	this->operators = operators;
	std::vector<int> mapped_operators = convert_symbols();
	std::sort(mapped_operators.begin(), mapped_operators.end());
	std::vector<int> ns(mapped_operators);
	ns.insert(ns.end(), numbers.begin(), numbers.end());
	this->tokens = ns;
}

std::vector<int> Expression::convert_symbols() {
	std::vector<int> res(5);
	int idx = 0;
	int val = 0;
	for (char const &c : operators) {
		switch (c) {
			case '+':
				val = -1;
				break;
			case '-':
				val = -2;
				break;
			case '*':
				val = -3;
				break;
			case '/':
				val = -4;
				break;
		}
		res[idx++] = val;
	}
	return res;
}

std::vector<int> Expression::evaluate_rpn() {

	std::vector<int> res;
	std::vector<int> stack;
	for (int token : tokens) {
		if (token > 0) {
			stack.push_back(token);
		} else {
			int b = stack.back();
			stack.pop_back();
			int a = stack.back();
			stack.pop_back();
			int val = 0;

			switch (token) {
				case -1: // addition
					val = a + b;
					break;
				case -2: // subtraction
					if (b > a) {
						res.push_back(-1);
						return res;
					}
					val = a - b;
					break;
				case -3: // multiplication
					val = a * b;
					break;
				case -4: // division
					if (b == 0 || a % b != 0) {
						res.push_back(-1);
						return res;
					}
					val = a / b;
					break;
			}
			stack.push_back(val);
			if (stack.size() == 1) {
				res.push_back(stack.back());
			}
		}
	}
	return res;
}

bool Expression::check_rpn() {
	int s = 0;
	for (int token : tokens) {
		int valence = 2 * (token < 0);
		s += 1 - valence;
		if (s <= 0) {
			return false;
		}
	}
	return true;
}

void Expression::evaluate(std::vector<int> &counts) {
	do {
		if (check_rpn()) {
			std::vector<int> vals = evaluate_rpn();
			for (int val : vals) {
				if (0 <= val && val < NUMBER_MAX_SIZE) {
					counts[val]++;
				}
			}
		}
	} while (std::next_permutation(tokens.begin(), tokens.end()));
}

std::vector<std::string> NumberSet::symbols;

void NumberSet::generate_symbols() {
	std::vector<std::string> res = {
		"+++++", "++++-", "++++*", "++++/", "+++--", "+++-*", "+++-/", 
		"+++**", "+++*/", "+++//", "++---", "++--*", "++--/", "++-**", 
		"++-*/", "++-//", "++***", "++**/", "++*//", "++///", "+----", 
		"+---*", "+---/", "+--**", "+--*/", "+--//", "+-***", "+-**/", 
		"+-*//", "+-///", "+****", "+***/", "+**//", "+*///", "+////", 
		"-----", "----*", "----/", "---**", "---*/", "---//", "--***", 
		"--**/", "--*//", "--///", "-****", "-***/", "-**//", "-*///", 
		"-////", "*****", "****/", "***//", "**///", "*////", "/////"
	};
	NumberSet::symbols = res;
}

std::vector<int> NumberSet::global_counts;

/**
 *	The NumberSet class represents a set of board numbers, without operators.
 *	There are 13243 different NumberSet instances generated, and each of them
 *	generates 56 Expression instances where the choices of operators are added.
 */

NumberSet::NumberSet(std::vector<int> numbers) {
	this->numbers = numbers;
	this->counts = std::vector<int>(NUMBER_MAX_SIZE);
	this->global_counts = std::vector<int>(NUMBER_MAX_SIZE);
}


void NumberSet::evaluate() {
	for (std::string s : NumberSet::symbols) {
		Expression e(numbers, s);
		e.evaluate(counts);
	}	
}

uint64_t sort_order(const std::vector<int> a) {
	uint64_t v = 0;
	for (int i : a) {
		v *= 101;
		v += i;
	}
	return v;
}

bool predicate(const NumberSet& a, const NumberSet& b) {
    return sort_order(a.numbers) < sort_order(b.numbers);
}

std::vector<NumberSet> NumberSet::generate_numbers() {
	int n = 24;
	int r = 6;
	std::set< std::vector<int> > combinations;

	std::vector<bool> v(n);
	std::fill(v.begin(), v.begin() + r, true);
	do {
		std::vector<int> nums(r);
		int idx = 0;
		for (int i = 0; i < n; ++i) {
			if (v[i]) {
				if (i < 20) {
					nums[idx++] = (i/2)+1; // small numbers (1-10)
				} else {
					nums[idx++] = (i-19)*25; // big numbers (25, 50, 75, 100)
				}
			}
		}
		combinations.insert(nums);
	} while (std::prev_permutation(v.begin(), v.end()));
	std::vector< NumberSet > numbers(
		combinations.begin(), combinations.end());
	std::sort(numbers.begin(), numbers.end(), predicate);
	return numbers;
}

std::ostream& operator<<(std::ostream &strm, const NumberSet &n) {
	for (int i = 0; i < 6; i++) {
		strm << n.numbers[i] << ",";
	}
	for (int i = 0; i < 999; i++) {
		strm << n.counts[i] << ",";
	}
	strm << n.counts[999];
	return strm;
}