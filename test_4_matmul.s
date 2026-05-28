	.text
	.globl	matmul_f64
	.type	matmul_f64, @function
matmul_f64:
entry:
	mv	a0, a0
	mv	a1, a1
	call	matmul_tile_f64
	mv	a2, a0
	mv	a0, a2
	ret
	.size	matmul_f64, .-matmul_f64
