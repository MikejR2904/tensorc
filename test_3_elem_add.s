	.text
	.globl	elem_add_f64
	.type	elem_add_f64, @function
elem_add_f64:
entry:
	vsetvli	t0, a0, 3
	vadd.vv	v10, v8, v9
	mv	a0, a0
	ret
	.size	elem_add_f64, .-elem_add_f64
