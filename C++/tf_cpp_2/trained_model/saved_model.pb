Ù
°
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
<
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
H
ShardedFilename
basename	
shard

num_shards
filename
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "train*1.3.02
b'unknown'÷
J
aPlaceholder*
_output_shapes
:*
shape:*
dtype0
z
W/W/Initializer/ConstConst*
_output_shapes
:*
valueB*  ?*
_class

loc:@W/W*
dtype0

W/W
VariableV2*
	container *
_class

loc:@W/W*
shared_name *
dtype0*
_output_shapes
:*
shape:


W/W/AssignAssignW/WW/W/Initializer/Const*
T0*
_output_shapes
:*
validate_shape(*
_class

loc:@W/W*
use_locking(
V
W/W/readIdentityW/W*
T0*
_output_shapes
:*
_class

loc:@W/W
:
bMulaW/W/read*
T0*
_output_shapes
:

initNoOp^W/W/Assign

init_1NoOp
P

save/ConstConst*
_output_shapes
: *
valueB Bmodel*
dtype0

save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_0a13faccf18e4dc995dcfd29445b2c1c/part*
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
\
save/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
d
save/SaveV2/tensor_namesConst*
_output_shapes
:*
valueBBW/W*
dtype0
e
save/SaveV2/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
y
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesW/W*
dtypes
2

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*

axis *
T0*
_output_shapes
:*
N
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
g
save/RestoreV2/tensor_namesConst*
_output_shapes
:*
valueBBW/W*
dtype0
h
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2

save/AssignAssignW/Wsave/RestoreV2*
T0*
_output_shapes
:*
validate_shape(*
_class

loc:@W/W*
use_locking(
(
save/restore_shardNoOp^save/Assign
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8":
trainable_variables#!

W/W:0
W/W/Assign
W/W/read:0"0
	variables#!

W/W:0
W/W/Assign
W/W/read:0