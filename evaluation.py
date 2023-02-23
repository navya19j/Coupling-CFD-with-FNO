from visualize import *

def eval(y_pred, y_true, maps,mask):
    # evaluate

    x_vals_pred, y_vals_pred, u_vals_pred = get_back(y_pred, maps,mask)
    x_vals_true, y_vals_true, u_vals_true = get_back(y_true, maps,None)

    # get where 0's in u_vals_true
    u_vals_true = np.array(u_vals_true)
    u_vals_pred = np.array(u_vals_pred)

    mask = u_vals_true == 0

    u_vals_true = u_vals_true[~mask]
    u_vals_pred = u_vals_pred[~mask]

    # rmse
    rmse = np.sqrt(np.mean((u_vals_true - u_vals_pred)**2))

    return rmse


