ÁĹ
˙
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
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
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02unknown8ĐÝ
d
count_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_13
]
count_13/Read/ReadVariableOpReadVariableOpcount_13*
_output_shapes
: *
dtype0
d
total_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_13
]
total_13/Read/ReadVariableOpReadVariableOptotal_13*
_output_shapes
: *
dtype0

training_16/SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nametraining_16/SGD/momentum
}
,training_16/SGD/momentum/Read/ReadVariableOpReadVariableOptraining_16/SGD/momentum*
_output_shapes
: *
dtype0

training_16/SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nametraining_16/SGD/learning_rate

1training_16/SGD/learning_rate/Read/ReadVariableOpReadVariableOptraining_16/SGD/learning_rate*
_output_shapes
: *
dtype0
~
training_16/SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nametraining_16/SGD/decay
w
)training_16/SGD/decay/Read/ReadVariableOpReadVariableOptraining_16/SGD/decay*
_output_shapes
: *
dtype0
|
training_16/SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *%
shared_nametraining_16/SGD/iter
u
(training_16/SGD/iter/Read/ReadVariableOpReadVariableOptraining_16/SGD/iter*
_output_shapes
: *
dtype0	
r
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_32/bias
k
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes
:*
dtype0
z
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_32/kernel
s
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes

:*
dtype0
r
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_31/bias
k
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes
:*
dtype0
z
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_31/kernel
s
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel*
_output_shapes

:*
dtype0

serving_default_dense_31_inputPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_31_inputdense_31/kerneldense_31/biasdense_32/kerneldense_32/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_4910

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ť
valueąBŽ B§

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures*
Ś
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
Ś
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
 
0
1
2
3*
 
0
1
2
3*
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
!trace_0
"trace_1
#trace_2
$trace_3* 
6
%trace_0
&trace_1
'trace_2
(trace_3* 
* 
:
)iter
	*decay
+learning_rate
,momentum*

-serving_default* 

0
1*

0
1*
* 

.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

3trace_0* 

4trace_0* 
_Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_31/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

:trace_0* 

;trace_0* 
_Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_32/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

<0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
WQ
VARIABLE_VALUEtraining_16/SGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEtraining_16/SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEtraining_16/SGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEtraining_16/SGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
H
=	variables
>	keras_api
	?total
	@count
A
_fn_kwargs*

?0
@1*

=	variables*
VP
VARIABLE_VALUEtotal_134keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcount_134keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¨
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp(training_16/SGD/iter/Read/ReadVariableOp)training_16/SGD/decay/Read/ReadVariableOp1training_16/SGD/learning_rate/Read/ReadVariableOp,training_16/SGD/momentum/Read/ReadVariableOptotal_13/Read/ReadVariableOpcount_13/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *&
f!R
__inference__traced_save_5053
Ű
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_31/kerneldense_31/biasdense_32/kerneldense_32/biastraining_16/SGD/itertraining_16/SGD/decaytraining_16/SGD/learning_ratetraining_16/SGD/momentumtotal_13count_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_restore_5093¤
Ë
Ř
G__inference_sequential_13_layer_call_and_return_conditional_losses_4899
dense_31_input*
dense_31_dense_31_kernel:$
dense_31_dense_31_bias:*
dense_32_dense_32_kernel:$
dense_32_dense_32_bias:
identity˘ dense_31/StatefulPartitionedCall˘ dense_32/StatefulPartitionedCall
 dense_31/StatefulPartitionedCallStatefulPartitionedCalldense_31_inputdense_31_dense_31_kerneldense_31_dense_31_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_31_layer_call_and_return_conditional_losses_4757¤
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0dense_32_dense_32_kerneldense_32_dense_32_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_32_layer_call_and_return_conditional_losses_4772x
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namedense_31_input
Ý
Đ
G__inference_sequential_13_layer_call_and_return_conditional_losses_4843

inputs*
dense_31_dense_31_kernel:$
dense_31_dense_31_bias:*
dense_32_dense_32_kernel:$
dense_32_dense_32_bias:
identity˘ dense_31/StatefulPartitionedCall˘ dense_32/StatefulPartitionedCall
 dense_31/StatefulPartitionedCallStatefulPartitionedCallinputsdense_31_dense_31_kerneldense_31_dense_31_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_31_layer_call_and_return_conditional_losses_4757¤
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0dense_32_dense_32_kerneldense_32_dense_32_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_32_layer_call_and_return_conditional_losses_4772x
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ą

˙
B__inference_dense_31_layer_call_and_return_conditional_losses_4757

inputs7
%matmul_readvariableop_dense_31_kernel:2
$biasadd_readvariableop_dense_31_bias:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_31_kernel*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_31_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


˙
B__inference_dense_31_layer_call_and_return_conditional_losses_4982

inputs7
%matmul_readvariableop_dense_31_kernel:2
$biasadd_readvariableop_dense_31_bias:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_31_kernel*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_31_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
š
í
,__inference_sequential_13_layer_call_fn_4784
dense_31_input!
dense_31_kernel:
dense_31_bias:!
dense_32_kernel:
dense_32_bias:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_31_inputdense_31_kerneldense_31_biasdense_32_kerneldense_32_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_13_layer_call_and_return_conditional_losses_4777o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namedense_31_input

ę
G__inference_sequential_13_layer_call_and_return_conditional_losses_4946

inputs@
.dense_31_matmul_readvariableop_dense_31_kernel:;
-dense_31_biasadd_readvariableop_dense_31_bias:@
.dense_32_matmul_readvariableop_dense_32_kernel:;
-dense_32_biasadd_readvariableop_dense_32_bias:
identity˘dense_31/BiasAdd/ReadVariableOp˘dense_31/MatMul/ReadVariableOp˘dense_32/BiasAdd/ReadVariableOp˘dense_32/MatMul/ReadVariableOp
dense_31/MatMul/ReadVariableOpReadVariableOp.dense_31_matmul_readvariableop_dense_31_kernel*
_output_shapes

:*
dtype0{
dense_31/MatMulMatMulinputs&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_31/BiasAdd/ReadVariableOpReadVariableOp-dense_31_biasadd_readvariableop_dense_31_bias*
_output_shapes
:*
dtype0
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
dense_31/ReluReludense_31/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_32/MatMul/ReadVariableOpReadVariableOp.dense_32_matmul_readvariableop_dense_32_kernel*
_output_shapes

:*
dtype0
dense_32/MatMulMatMuldense_31/Relu:activations:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_32/BiasAdd/ReadVariableOpReadVariableOp-dense_32_biasadd_readvariableop_dense_32_bias*
_output_shapes
:*
dtype0
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
dense_32/SoftmaxSoftmaxdense_32/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentitydense_32/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ě
NoOpNoOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ż
 
'__inference_dense_32_layer_call_fn_4989

inputs!
dense_32_kernel:
dense_32_bias:
identity˘StatefulPartitionedCallć
StatefulPartitionedCallStatefulPartitionedCallinputsdense_32_kerneldense_32_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_32_layer_call_and_return_conditional_losses_4772o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ę
G__inference_sequential_13_layer_call_and_return_conditional_losses_4964

inputs@
.dense_31_matmul_readvariableop_dense_31_kernel:;
-dense_31_biasadd_readvariableop_dense_31_bias:@
.dense_32_matmul_readvariableop_dense_32_kernel:;
-dense_32_biasadd_readvariableop_dense_32_bias:
identity˘dense_31/BiasAdd/ReadVariableOp˘dense_31/MatMul/ReadVariableOp˘dense_32/BiasAdd/ReadVariableOp˘dense_32/MatMul/ReadVariableOp
dense_31/MatMul/ReadVariableOpReadVariableOp.dense_31_matmul_readvariableop_dense_31_kernel*
_output_shapes

:*
dtype0{
dense_31/MatMulMatMulinputs&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_31/BiasAdd/ReadVariableOpReadVariableOp-dense_31_biasadd_readvariableop_dense_31_bias*
_output_shapes
:*
dtype0
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
dense_31/ReluReludense_31/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_32/MatMul/ReadVariableOpReadVariableOp.dense_32_matmul_readvariableop_dense_32_kernel*
_output_shapes

:*
dtype0
dense_32/MatMulMatMuldense_31/Relu:activations:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_32/BiasAdd/ReadVariableOpReadVariableOp-dense_32_biasadd_readvariableop_dense_32_bias*
_output_shapes
:*
dtype0
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
dense_32/SoftmaxSoftmaxdense_32/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentitydense_32/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ě
NoOpNoOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ś

˙
B__inference_dense_32_layer_call_and_return_conditional_losses_4772

inputs7
%matmul_readvariableop_dense_32_kernel:2
$biasadd_readvariableop_dense_32_bias:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_32_kernel*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_32_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ă
"__inference_signature_wrapper_4910
dense_31_input!
dense_31_kernel:
dense_31_bias:!
dense_32_kernel:
dense_32_bias:
identity˘StatefulPartitionedCallď
StatefulPartitionedCallStatefulPartitionedCalldense_31_inputdense_31_kerneldense_31_biasdense_32_kerneldense_32_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_4739o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namedense_31_input


˙
B__inference_dense_32_layer_call_and_return_conditional_losses_5000

inputs7
%matmul_readvariableop_dense_32_kernel:2
$biasadd_readvariableop_dense_32_bias:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_32_kernel*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_32_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ý
Đ
G__inference_sequential_13_layer_call_and_return_conditional_losses_4777

inputs*
dense_31_dense_31_kernel:$
dense_31_dense_31_bias:*
dense_32_dense_32_kernel:$
dense_32_dense_32_bias:
identity˘ dense_31/StatefulPartitionedCall˘ dense_32/StatefulPartitionedCall
 dense_31/StatefulPartitionedCallStatefulPartitionedCallinputsdense_31_dense_31_kerneldense_31_dense_31_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_31_layer_call_and_return_conditional_losses_4757¤
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0dense_32_dense_32_kerneldense_32_dense_32_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_32_layer_call_and_return_conditional_losses_4772x
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ż
 
'__inference_dense_31_layer_call_fn_4971

inputs!
dense_31_kernel:
dense_31_bias:
identity˘StatefulPartitionedCallć
StatefulPartitionedCallStatefulPartitionedCallinputsdense_31_kerneldense_31_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_31_layer_call_and_return_conditional_losses_4757o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ą
ĺ
,__inference_sequential_13_layer_call_fn_4928

inputs!
dense_31_kernel:
dense_31_bias:!
dense_32_kernel:
dense_32_bias:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsdense_31_kerneldense_31_biasdense_32_kerneldense_32_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_13_layer_call_and_return_conditional_losses_4843o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ë
ş
__inference__wrapped_model_4739
dense_31_inputN
<sequential_13_dense_31_matmul_readvariableop_dense_31_kernel:I
;sequential_13_dense_31_biasadd_readvariableop_dense_31_bias:N
<sequential_13_dense_32_matmul_readvariableop_dense_32_kernel:I
;sequential_13_dense_32_biasadd_readvariableop_dense_32_bias:
identity˘-sequential_13/dense_31/BiasAdd/ReadVariableOp˘,sequential_13/dense_31/MatMul/ReadVariableOp˘-sequential_13/dense_32/BiasAdd/ReadVariableOp˘,sequential_13/dense_32/MatMul/ReadVariableOpŠ
,sequential_13/dense_31/MatMul/ReadVariableOpReadVariableOp<sequential_13_dense_31_matmul_readvariableop_dense_31_kernel*
_output_shapes

:*
dtype0
sequential_13/dense_31/MatMulMatMuldense_31_input4sequential_13/dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ľ
-sequential_13/dense_31/BiasAdd/ReadVariableOpReadVariableOp;sequential_13_dense_31_biasadd_readvariableop_dense_31_bias*
_output_shapes
:*
dtype0ť
sequential_13/dense_31/BiasAddBiasAdd'sequential_13/dense_31/MatMul:product:05sequential_13/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙~
sequential_13/dense_31/ReluRelu'sequential_13/dense_31/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
,sequential_13/dense_32/MatMul/ReadVariableOpReadVariableOp<sequential_13_dense_32_matmul_readvariableop_dense_32_kernel*
_output_shapes

:*
dtype0ş
sequential_13/dense_32/MatMulMatMul)sequential_13/dense_31/Relu:activations:04sequential_13/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ľ
-sequential_13/dense_32/BiasAdd/ReadVariableOpReadVariableOp;sequential_13_dense_32_biasadd_readvariableop_dense_32_bias*
_output_shapes
:*
dtype0ť
sequential_13/dense_32/BiasAddBiasAdd'sequential_13/dense_32/MatMul:product:05sequential_13/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
sequential_13/dense_32/SoftmaxSoftmax'sequential_13/dense_32/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
IdentityIdentity(sequential_13/dense_32/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp.^sequential_13/dense_31/BiasAdd/ReadVariableOp-^sequential_13/dense_31/MatMul/ReadVariableOp.^sequential_13/dense_32/BiasAdd/ReadVariableOp-^sequential_13/dense_32/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 2^
-sequential_13/dense_31/BiasAdd/ReadVariableOp-sequential_13/dense_31/BiasAdd/ReadVariableOp2\
,sequential_13/dense_31/MatMul/ReadVariableOp,sequential_13/dense_31/MatMul/ReadVariableOp2^
-sequential_13/dense_32/BiasAdd/ReadVariableOp-sequential_13/dense_32/BiasAdd/ReadVariableOp2\
,sequential_13/dense_32/MatMul/ReadVariableOp,sequential_13/dense_32/MatMul/ReadVariableOp:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namedense_31_input
š
í
,__inference_sequential_13_layer_call_fn_4879
dense_31_input!
dense_31_kernel:
dense_31_bias:!
dense_32_kernel:
dense_32_bias:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_31_inputdense_31_kerneldense_31_biasdense_32_kerneldense_32_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_13_layer_call_and_return_conditional_losses_4843o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namedense_31_input
Ą
ĺ
,__inference_sequential_13_layer_call_fn_4919

inputs!
dense_31_kernel:
dense_31_bias:!
dense_32_kernel:
dense_32_bias:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsdense_31_kerneldense_31_biasdense_32_kerneldense_32_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_13_layer_call_and_return_conditional_losses_4777o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ů
Ú
__inference__traced_save_5053
file_prefix.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop3
/savev2_training_16_sgd_iter_read_readvariableop	4
0savev2_training_16_sgd_decay_read_readvariableop<
8savev2_training_16_sgd_learning_rate_read_readvariableop7
3savev2_training_16_sgd_momentum_read_readvariableop'
#savev2_total_13_read_readvariableop'
#savev2_count_13_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ľ
valueŤB¨B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop/savev2_training_16_sgd_iter_read_readvariableop0savev2_training_16_sgd_decay_read_readvariableop8savev2_training_16_sgd_learning_rate_read_readvariableop3savev2_training_16_sgd_momentum_read_readvariableop#savev2_total_13_read_readvariableop#savev2_count_13_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*C
_input_shapes2
0: ::::: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
+

 __inference__traced_restore_5093
file_prefix2
 assignvariableop_dense_31_kernel:.
 assignvariableop_1_dense_31_bias:4
"assignvariableop_2_dense_32_kernel:.
 assignvariableop_3_dense_32_bias:1
'assignvariableop_4_training_16_sgd_iter:	 2
(assignvariableop_5_training_16_sgd_decay: :
0assignvariableop_6_training_16_sgd_learning_rate: 5
+assignvariableop_7_training_16_sgd_momentum: %
assignvariableop_8_total_13: %
assignvariableop_9_count_13: 
identity_11˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ľ
valueŤB¨B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B Ő
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_31_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_31_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_32_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_32_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_4AssignVariableOp'assignvariableop_4_training_16_sgd_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp(assignvariableop_5_training_16_sgd_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp0assignvariableop_6_training_16_sgd_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp+assignvariableop_7_training_16_sgd_momentumIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_total_13Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_count_13Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ť
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ë
Ř
G__inference_sequential_13_layer_call_and_return_conditional_losses_4889
dense_31_input*
dense_31_dense_31_kernel:$
dense_31_dense_31_bias:*
dense_32_dense_32_kernel:$
dense_32_dense_32_bias:
identity˘ dense_31/StatefulPartitionedCall˘ dense_32/StatefulPartitionedCall
 dense_31/StatefulPartitionedCallStatefulPartitionedCalldense_31_inputdense_31_dense_31_kerneldense_31_dense_31_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_31_layer_call_and_return_conditional_losses_4757¤
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0dense_32_dense_32_kerneldense_32_dense_32_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_32_layer_call_and_return_conditional_losses_4772x
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namedense_31_input"ľ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*š
serving_defaultĽ
I
dense_31_input7
 serving_default_dense_31_input:0˙˙˙˙˙˙˙˙˙<
dense_320
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:ŕV
´
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures"
_tf_keras_sequential
ť
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ť
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
non_trainable_variables

layers
metrics
layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ĺ
!trace_0
"trace_1
#trace_2
$trace_32ú
,__inference_sequential_13_layer_call_fn_4784
,__inference_sequential_13_layer_call_fn_4919
,__inference_sequential_13_layer_call_fn_4928
,__inference_sequential_13_layer_call_fn_4879ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z!trace_0z"trace_1z#trace_2z$trace_3
Ń
%trace_0
&trace_1
'trace_2
(trace_32ć
G__inference_sequential_13_layer_call_and_return_conditional_losses_4946
G__inference_sequential_13_layer_call_and_return_conditional_losses_4964
G__inference_sequential_13_layer_call_and_return_conditional_losses_4889
G__inference_sequential_13_layer_call_and_return_conditional_losses_4899ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z%trace_0z&trace_1z'trace_2z(trace_3
ŃBÎ
__inference__wrapped_model_4739dense_31_input"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
I
)iter
	*decay
+learning_rate
,momentum"
	optimizer
,
-serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ë
3trace_02Î
'__inference_dense_31_layer_call_fn_4971˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z3trace_0

4trace_02é
B__inference_dense_31_layer_call_and_return_conditional_losses_4982˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z4trace_0
!:2dense_31/kernel
:2dense_31/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ë
:trace_02Î
'__inference_dense_32_layer_call_fn_4989˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z:trace_0

;trace_02é
B__inference_dense_32_layer_call_and_return_conditional_losses_5000˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z;trace_0
!:2dense_32/kernel
:2dense_32/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_sequential_13_layer_call_fn_4784dense_31_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ýBú
,__inference_sequential_13_layer_call_fn_4919inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ýBú
,__inference_sequential_13_layer_call_fn_4928inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
,__inference_sequential_13_layer_call_fn_4879dense_31_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
G__inference_sequential_13_layer_call_and_return_conditional_losses_4946inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
G__inference_sequential_13_layer_call_and_return_conditional_losses_4964inputs"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 B
G__inference_sequential_13_layer_call_and_return_conditional_losses_4889dense_31_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 B
G__inference_sequential_13_layer_call_and_return_conditional_losses_4899dense_31_input"ż
ś˛˛
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
:	 (2training_16/SGD/iter
: (2training_16/SGD/decay
':% (2training_16/SGD/learning_rate
":  (2training_16/SGD/momentum
ĐBÍ
"__inference_signature_wrapper_4910dense_31_input"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŰBŘ
'__inference_dense_31_layer_call_fn_4971inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
öBó
B__inference_dense_31_layer_call_and_return_conditional_losses_4982inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŰBŘ
'__inference_dense_32_layer_call_fn_4989inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
öBó
B__inference_dense_32_layer_call_and_return_conditional_losses_5000inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
^
=	variables
>	keras_api
	?total
	@count
A
_fn_kwargs"
_tf_keras_metric
.
?0
@1"
trackable_list_wrapper
-
=	variables"
_generic_user_object
:  (2total_13
:  (2count_13
 "
trackable_dict_wrapper
__inference__wrapped_model_4739t7˘4
-˘*
(%
dense_31_input˙˙˙˙˙˙˙˙˙
Ş "3Ş0
.
dense_32"
dense_32˙˙˙˙˙˙˙˙˙˘
B__inference_dense_31_layer_call_and_return_conditional_losses_4982\/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 z
'__inference_dense_31_layer_call_fn_4971O/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙˘
B__inference_dense_32_layer_call_and_return_conditional_losses_5000\/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 z
'__inference_dense_32_layer_call_fn_4989O/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙š
G__inference_sequential_13_layer_call_and_return_conditional_losses_4889n?˘<
5˘2
(%
dense_31_input˙˙˙˙˙˙˙˙˙
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 š
G__inference_sequential_13_layer_call_and_return_conditional_losses_4899n?˘<
5˘2
(%
dense_31_input˙˙˙˙˙˙˙˙˙
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ą
G__inference_sequential_13_layer_call_and_return_conditional_losses_4946f7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ą
G__inference_sequential_13_layer_call_and_return_conditional_losses_4964f7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 
,__inference_sequential_13_layer_call_fn_4784a?˘<
5˘2
(%
dense_31_input˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
,__inference_sequential_13_layer_call_fn_4879a?˘<
5˘2
(%
dense_31_input˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
,__inference_sequential_13_layer_call_fn_4919Y7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
,__inference_sequential_13_layer_call_fn_4928Y7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙­
"__inference_signature_wrapper_4910I˘F
˘ 
?Ş<
:
dense_31_input(%
dense_31_input˙˙˙˙˙˙˙˙˙"3Ş0
.
dense_32"
dense_32˙˙˙˙˙˙˙˙˙