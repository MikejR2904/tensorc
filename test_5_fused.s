	.text
	.globl	matmul_relu_f64
	.type	matmul_relu_f64, @function
matmul_relu_f64:
entry:
	mv	a0, a0
	mv	a1, a1
	call	matmul_relu_tile_f64
	mv	a2, a0
	mv	a0, a2
	ret
	.size	matmul_relu_f64, .-matmul_relu_f64
