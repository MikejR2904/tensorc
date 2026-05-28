	.text
	.globl	add_i64
	.type	add_i64, @function
add_i64:
entry:
	add	a2, a0, a1
	mv	a0, a2
	ret
	.size	add_i64, .-add_i64
