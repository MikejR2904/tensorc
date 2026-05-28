	.text
	.globl	mul_i64
	.type	mul_i64, @function
mul_i64:
entry:
	mul	a2, a0, a1
	mv	a0, a2
	ret
	.size	mul_i64, .-mul_i64
