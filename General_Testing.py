import numpy as np
import io
import sys

RTOL=1e-05 
ATOL=1e-05 

def sub_get_difference_string(pred, ground, index_list, atol, diff_string):
    if len(pred.shape) == 1:
        for i in range(pred.shape[0]):
            diff = np.absolute(float(pred[i]) - float(ground[i]))
            if diff >= atol:
                temp_list = index_list + [i]
                diff_string += str(temp_list) + " :\t" + str(pred[i]) + "\tvs\t" + str(ground[i]) + "\n"
    else:
        for i in range(pred.shape[0]):
            temp_list = index_list + [i]
            diff_string = sub_get_difference_string(pred[i], ground[i], temp_list, atol, diff_string)            
    return diff_string

def get_difference_string(pred, ground, atol):
    diff_string = ""
    if pred.shape == ground.shape:
        diff_string = "INDEX\tPRED\t\tGROUND\n"
        diff_string = sub_get_difference_string(pred, ground, [], atol, diff_string)
    return diff_string

def check_for_unequal(base_error_msg, filename, pred, ground):
    try:
        error_msg = base_error_msg + ": " + filename
        np.testing.assert_allclose(pred, ground, rtol=RTOL, atol=ATOL, 
                                err_msg=error_msg,
                                verbose=True)
    except AssertionError as ae:                        
        diff_string = get_difference_string(pred, ground, ATOL)
        raise AssertionError("\n" + error_msg + "\n" + diff_string) from ae
    
def check_for_unequal_no_diff(base_error_msg, filename, pred, ground):
    error_msg = base_error_msg + ": " + filename
    np.testing.assert_allclose(pred, ground, rtol=RTOL, atol=ATOL, 
                            err_msg=error_msg,
                            verbose=True)  
 
def redirect_std_out():
    cap_stream = io.StringIO()   
    sys.stdout = cap_stream  
    return cap_stream

def restore_std_out():
    sys.stdout = sys.__stdout__                   
    
def get_captured_output(cap_stream):
    return cap_stream.getvalue()
  