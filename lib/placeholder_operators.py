from lib.basic_classes import DeferredTensorOp

## Define tensorial operators

class inner_prod_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'inner_prod_op'.
    pass

class norm_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'norm_op'.
    pass


class dot_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'dot_op'.
    pass
    
        
class contract_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'contract_op'.
    pass

class doublecontract_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'doublecontract_op'.
    pass
        
class grad_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'doublecontract_op'.
    pass

class symgrad_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'doublecontract_op'.
    pass

class div_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'div_op'.
    pass

class curl_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp
    # which uses the YAML definition for 'curl_op'.
    pass

class curl_3d_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp
    # which uses the YAML definition for 'curl_3d_op'.
    pass
        
class curl_2d_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp
    # which uses the YAML definition for 'curl_2d_op'.
    pass

class matrix_prod_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'matrix_prod_op'.
    pass
        
class matrix_transpose_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'matrix_transpose_op'.
    pass

class matrix_vector_prod_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'matrix_vector_prod_op'.
    pass
        
class matrix_det_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'matrix_det_op'.
    pass
        
class matrix_inv_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'matrix_inv_op'.
    pass
        
class matrix_cofactor_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'matrix_cofactor_op'.
    pass

class vector_cross_prod_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'vector_cross_prod_op'.
    pass

class vector_cross_prod_3d_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'vector_cross_prod_3d_op'.
    pass

class vector_cross_prod_2d_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'vector_cross_prod_2d_op'.
    pass

class vector_outer_prod_op(DeferredTensorOp):
    # It inherits all behavior from DeferredTensorOp 
    # which uses the YAML definition for 'vector_outer_prod_op'.
    pass

# class flatten_and_combine_tensors(sp.Function):
#     precedence = PRECEDENCE_TRADITIONAL['Dot'] # Probably wrong

#     @classmethod
#     def eval(cls, a):
#         if not a.has(BaseTensorPlaceholder):
#             return tensor_arrays_flatten_and_combine(a)
        
#     def _latex(self, printer, exp=None, *args):
#         if isinstance(printer, CustomLatexPrinter):
#             a = self.args[0]
#             rv = r"%s" % printer.doprint(a)
#             if exp is None:
#                 return rv
#             else:
#                 return r"%s^{%s}" % (rv, exp)
#         else:
#             return printer._print_Function(self, exp=exp)