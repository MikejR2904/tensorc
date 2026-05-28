	.text
	.globl	max_i64
	.type	max_i64, @function
max_i64:
entry:
	slt	a2, a0, a1
	bnez	a2, true_block
	j	false_block
true_block:
	mv	a0, a1
	ret
false_block:
	mv	a0, a0
	ret
merge_block:
	.size	max_i64, .-max_i64
