#include "Autograd.h"

int main()
{
	printf("Here\n");
	Value a = CreateNewValue(1.0);
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
