#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// FP32
#define FIXEDPT_BITS 32
#define FIXEDPT_WBITS 9
#include "FixedPoint.h"
float relu_alpha = 0.01;
enum OperationEnum{ADD,SUBTRACT,MULTIPLY,DIVIDE,POWER,OPERATION_LENGTH};
typedef struct value_struct *Value;

struct value_struct
{
	float number;
	float gradient;
	fixedpt numberInteger;
	fixedpt gradientInteger;
	Value children[2];
	int numberOfChildren;
	int operationIndex;
	void (*backward)(struct value_struct*);
};

Value CreateNewValue(float value)
{
	Value newValue = malloc(sizeof(*newValue));
	newValue->number = value;
	newValue->numberInteger = fixedpt_rconst(value);
	newValue->gradientInteger = 0;
	newValue->gradient = 0.0f;
	newValue->children[0] = NULL;
	newValue->children[1] = NULL;
	newValue->numberOfChildren = 0;
	newValue->operationIndex = 0;
	newValue->backward = NULL;
	return newValue;
}

void PrintValue(Value value)
{
	if(value != NULL)
	{
		printf("Value(value = %.3f, gradient = %.2f)\n", value->number, value->gradient);
	}
}

void PrintValueInteger(Value value)
{
	if(value != NULL)
	{
		printf("Value(value = %d, gradient = %d)\n", value->numberInteger, value->gradientInteger);
	}
}


void DestroyValue(Value value)
{
	free(value);
}

void ClipGradient(Value v, float minimumValue, float maximumValue){if(v->gradient < minimumValue){v->gradient = minimumValue;}else if(v->gradient > maximumValue) {v->gradient = maximumValue;}}
void AddBack(Value v)
{
	v->children[0]->gradient += v->gradient;
	v->children[1]->gradient += v->gradient;
	v->children[0]->gradientInteger = fixedpt_rconst(v->children[0]->gradient);
	v->children[1]->gradientInteger = fixedpt_rconst(v->children[1]->gradient);
	//ClipGradient(v->children[0], -10.0, 10.0);
	//ClipGradient(v->children[1], -10.0, 10.0);
}

void SubtractBack(Value v)
{
	v->children[0]->gradient += v->gradient;
	v->children[1]->gradient -= v->gradient;
	
	v->children[0]->gradientInteger = fixedpt_rconst(v->children[0]->gradient);
	v->children[1]->gradientInteger = fixedpt_rconst(v->children[1]->gradient);
	//ClipGradient(v->children[0], -10.0, 10.0);
	//ClipGradient(v->children[1], -10.0, 10.0);
}

void MultiplyBack(Value v)
{
	v->children[0]->gradient += v->children[1]->number * v->gradient;
	v->children[1]->gradient += v->children[0]->number * v->gradient;
	v->children[0]->gradientInteger = fixedpt_rconst(v->children[0]->gradient);
	v->children[1]->gradientInteger = fixedpt_rconst(v->children[1]->gradient);
	
	//ClipGradient(v->children[0], -10.0, 10.0);
	//ClipGradient(v->children[1], -10.0, 10.0);
}

void DivideBack(Value v)
{
	v->children[0]->gradient += (1.0 / v->children[1]->number) * v->gradient;
	v->children[1]->gradient += (-v->children[0]->number / (v->children[1]->number * v->children[1]->number)) * v->gradient;
	v->children[0]->gradientInteger = fixedpt_rconst(v->children[0]->gradient);
	v->children[1]->gradientInteger = fixedpt_rconst(v->children[1]->gradient);
	
	//ClipGradient(v->children[0], -10.0, 10.0);
	//ClipGradient(v->children[1], -10.0, 10.0);
}

void PowerBack(Value v)
{
	v->children[0]->gradient += (1.0 / v->children[1]->number) * v->gradient;
	v->children[1]->gradient += (-v->children[0]->number / (v->children[1]->number * v->children[1]->number)) * v->gradient;
	
	v->children[0]->gradient += (v->children[1]->number * pow(v->children[0]->number, v->children[1]->number - 1)) * v->gradient;
	if(v->children[0]->number > 0) 
	{
		// Ensure base is positive before computing log
		v->children[1]->gradient += (log(v->children[0]->number) * pow(v->children[0]->number, v->children[1]->number)) * v->gradient;
	}
	v->children[0]->gradientInteger = fixedpt_rconst(v->children[0]->gradient);
	v->children[1]->gradientInteger = fixedpt_rconst(v->children[1]->gradient);
	
	//ClipGradient(v->children[0], -10.0, 10.0);
	//ClipGradient(v->children[1], -10.0, 10.0);
}

void ReluBack(Value v)
{
	if(v->children[0]->number > 0)
	{
		v->children[0]->gradient += v->gradient;
	}
	else
	{
		v->children[0]->gradient += v->gradient * relu_alpha;
	}
	v->children[0]->gradientInteger = fixedpt_rconst(v->children[0]->gradient);	
}

Value AddValues(Value a, Value b)
{
	Value value = CreateNewValue(a->number + b->number);
	value->children[0] = a;
	value->children[1] = b;
	value->numberOfChildren = 2;
	value->operationIndex = ADD;
	value->backward = AddBack;
	return value;
}

Value SubtractValues(Value a, Value b)
{
	Value value = CreateNewValue(a->number - b->number);
	value->children[0] = a;
	value->children[1] = b;
	value->numberOfChildren = 2;
	value->operationIndex = SUBTRACT;
	value->backward = SubtractBack;
	return value;
}

Value MultiplyValues(Value a, Value b)
{
	Value value = CreateNewValue(a->number * b->number);
	value->children[0] = a;
	value->children[1] = b;
	value->numberOfChildren = 2;
	value->operationIndex = MULTIPLY;
	value->backward = MultiplyBack;
	return value;
}

Value DivideValues(Value a, Value b)
{
	assert(b->number != 0.0f);
	Value value = CreateNewValue(a->number / b->number);
	value->children[0] = a;
	value->children[1] = b;
	value->numberOfChildren = 2;
	value->operationIndex = DIVIDE;
	value->backward = DivideBack;
	return value;
}

Value PowerValues(Value a, Value b)
{
	assert(b->number != 0.0f);
	Value value = CreateNewValue(pow(a->number , b->number));
	value->children[0] = a;
	value->children[1] = b;
	value->numberOfChildren = 2;
	value->operationIndex = DIVIDE;
	value->backward = PowerBack;
	return value;
}

Value ReluValue(Value a)
{
	Value value = CreateNewValue(0.0f);
	if(a->number > 0.0f)
	{
		value->number = a->number;
	}
	else
	{
		value->number = relu_alpha * a->number;
	}
	value->gradient = 0.0f;
	value->children[0] = a;
	value->children[1] = NULL;
	value->numberOfChildren = 1;
	value->backward = ReluBack;
	return value;
}




void build_topo(Value v, Value* topo, int* topo_size, Value* visited, int* visited_size)
{
	for(int i = 0; i < *visited_size; ++i){if (visited[i] == v) return;}
	visited[*visited_size] = v;
	(*visited_size)++;
	for(int i = 0; i < v->numberOfChildren; ++i)
	{
		build_topo(v->children[i], topo, topo_size, visited, visited_size);
	}
	topo[*topo_size] = v;
	(*topo_size)++;
}

void backward(Value root)
{
    Value topo[1000];  // Assuming a maximum of 100 nodes in the computation graph for simplicity
    int topo_size = 0;
    Value visited[1000];
    int visited_size = 0;

    build_topo(root, topo, &topo_size, visited, &visited_size);

    root->gradient = 1.0;

    for (int i = topo_size - 1; i >= 0; --i) {
        // printf("%.2f", topo[i]->val);
        // printf("\n");
        if (topo[i]->backward) {
            topo[i]->backward(topo[i]);
        }
    }
}
typedef Value (*f)(Value, Value);  
f ForwardOps[] = {&AddValues, &SubtractValues, &MultiplyValues, &DivideValues};

typedef void (*g)(Value);  
g BackwardOps[] = {&AddBack, &SubtractBack, &MultiplyBack, &DivideBack, &PowerBack};



/*
#include "Autograd.h"

int main()
{
	Value a = CreateNewValue(2.0);
	Value b = CreateNewValue(-3.0);
	Value c = CreateNewValue(9.0);
	Value e = ForwardOps[MULTIPLY](a,b); 
	Value d = ForwardOps[ADD](e,c); 
	Value f = CreateNewValue(-2.0);
	Value L = ForwardOps[MULTIPLY](d,f);
	backward(L);
	PrintValue(a);
	PrintValue(b);
	PrintValue(c);
	PrintValue(d);
	PrintValue(e);
	PrintValue(f);
	PrintValue(L);
	DestroyValue(a);
	DestroyValue(b);
	DestroyValue(c);
	DestroyValue(d);
	DestroyValue(e);
	DestroyValue(f);
	DestroyValue(L);
	return 0;
}
*/

