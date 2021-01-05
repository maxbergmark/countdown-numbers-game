#define NUM_TOKENS 11
#define MAX_TARGET 1000

#define NUM_EXTRA_VALUES 3
#define SUBTRACTION_FAIL_INDEX 1000
#define DIVISION_FAIL_INDEX 1001
#define PERMUTATION_FAIL_INDEX 1002

void swap(int* a, int* b) {
	int tmp = *a;
	*a = *b;
	*b = tmp;
}

void reverse(int* first, int* last) {	
	for (; first != last && first != --last; ++first)
		swap(first, last);
}

void next_permutation(int* first, int* last) {
	int* next = last;
	--next;
	if(first == last || first == next) {
		return;
	}

	while(true)
	{
		int* next1 = next;
		--next;
		if(*next < *next1)
		{
			int* mid = last;
			--mid;
			for(; !(*next < *mid); --mid)
				;
			swap(next, mid);
			reverse(next1, last);
			return;
		}

		if(next == first)
		{
			reverse(first, last);
			return;
		}
	}
}

void print(int nums[], int n) {
	for (int i = 0; i < n; i++) {
		printf("%d ", nums[i]);
	}
	printf("\n");
}

void evaluate_rpn(int tokens[NUM_TOKENS], int res[10], int n, int *res_n, 
	int local_result[MAX_TARGET + NUM_EXTRA_VALUES]) {

	int stack[10];
	int stack_i = 0;
	int res_i = 0;

	for (int i = 0; i < n; i++) {
		int token = tokens[i];
		if (token > 0) {
			stack[stack_i++] = token;
		} else {
			int b = stack[--stack_i];
			int a = stack[--stack_i];
			int val = 0;

			switch (token) {
				case -1: // addition
					val = a + b;
					break;
				case -2: // subtraction
					if (b > a) {
						*res_n = res_i;
						local_result[SUBTRACTION_FAIL_INDEX]++;
						return;
					}
					val = a - b;
					break;
				case -3: // multiplication
					val = a * b;
					break;
				case -4: // division
					if (b == 0 || a % b != 0) {
						*res_n = res_i;
						local_result[DIVISION_FAIL_INDEX]++;
						return;
					}
					val = a / b;
					break;
			}
			stack[stack_i++] = val;
			if (stack_i == 1) {
				res[res_i++] = stack[0];
			}
		}
	}
	*res_n = res_i;
	return;
}

bool check_rpn(int tokens[NUM_TOKENS], int n) {
	int s = 0;
	for (int i = 0; i < n; i++) {
		int token = tokens[i];
		int valence = 2 * (token < 0);
		s += 1 - valence;
		if (s <= 0) {
			return false;
		}
	}
	return true;
}

int fac(int n) {
	int p = 1;
	for (int i = 1; i <= n; i++) {
		p *= i;
	}
	return p;
}

kernel void evaluate(__global int *data_g, __global int *result, 
	__constant int *dims) {
	
	int tid = get_global_id(0);
	int idx = dims[1] * tid;
	int local_data[NUM_TOKENS];
	for (int i = 0; i < NUM_TOKENS; i++) {
		local_data[i] = data_g[i + idx];
	}
	
	int *first = &local_data[0];
	int *last = &local_data[NUM_TOKENS];
	int limit = dims[2];
	int rpn_res[10];
	int rpn_n;
	
	int local_result[MAX_TARGET + NUM_EXTRA_VALUES];
	for (int i = 0; i < MAX_TARGET + NUM_EXTRA_VALUES; i++) {
		local_result[i] = 0;
	}

	for (int i = 0; i < limit; i++) {
		if (check_rpn(local_data, NUM_TOKENS)) {
			evaluate_rpn(local_data, rpn_res, NUM_TOKENS, &rpn_n, local_result);
			for (int j = 0; j < rpn_n; j++) {
				int val = rpn_res[j];
				int in_range = 0 <= val & val < MAX_TARGET;
				local_result[val * in_range] += in_range;
			}
		} else {
			local_result[PERMUTATION_FAIL_INDEX]++;
		}
		next_permutation(first, last);
	}

	for (int i = 0; i < MAX_TARGET + NUM_EXTRA_VALUES; i++) {
		result[(MAX_TARGET + NUM_EXTRA_VALUES) * tid + i] = local_result[i];
	}

}
