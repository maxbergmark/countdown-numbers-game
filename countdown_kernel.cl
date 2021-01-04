// function to swap character 
// a - the character to swap with b
// b - the character to swap with a
void swap(int* a, int* b) {
	int tmp = *a;
	*a = *b;
	*b = tmp;
}


// function to reverse the array (sub array in array)
// first - 1st character in the array (sub-array in array)
// last - 1 character past the last character
void reverse(int* first, int* last) {	
	for (; first != last && first != --last; ++first)
		swap(first, last);
}


// function to find the next permutation (sub array in array)
// first - 1st character in the array (sub-array in array)
// last - 1 character past the last character
void next_permutation(int* first, int* last) {
	int* next = last;
	--next;
	if(first == last || first == next)
		return;

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

void evaluate_rpn(int tokens[11], int res[10], int n, int *res_n) {

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
						// res[res_i++] = -1;
						*res_n = res_i-0;
						return;
					}
					val = a - b;
					break;
				case -3: // multiplication
					val = a * b;
					break;
				case -4: // division
					if (b == 0 || a % b != 0) {
						// res[res_i++] = -1;
						*res_n = res_i-0;
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

bool check_rpn(int tokens[11], int n) {
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
	int local_data[11];
	for (int i = 0; i < 11; i++) {
		local_data[i] = data_g[i + idx];
	}
	
	int *first = &local_data[0];
	int *last = &local_data[11];
	int limit = dims[2];
	int rpn_res[10];
	int rpn_n;
	
	int local_result[1000];
	for (int i = 0; i < 1000; i++) {
		local_result[i] = 0;
	}

	for (int i = 0; i < limit; i++) {
		if (check_rpn(local_data, 11)) {
			evaluate_rpn(local_data, rpn_res, 11, &rpn_n);
			for (int j = 0; j < rpn_n; j++) {
				int val = rpn_res[j];
				int in_range = 0 <= val & val < 1000;
				local_result[val * in_range] += in_range;
			}
		}
		next_permutation(first, last);
	}

	for (int i = 0; i < 1000; i++) {
		result[1000 * tid + i] = local_result[i];
	}

}
