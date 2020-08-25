# Optimization utilities for graph neural networks

import time
import itertools
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.nn import init
from torch_geometric.nn import GNNExplainer
from sig.utils import io_utils
from sig.utils import graph_utils
from sig.nn.loss import classification_loss as class_loss
from sig.nn.loss import regularization_loss as reg_loss
from sig.nn.models.gat import GAT
from sig.nn.models.gcn import GCN
from sig.nn.models.sigcn import SIGCN


def train(
    model,
    x,
    edge_index,
    y,
    mask,
    optimizer,
    sig_coeffs=None,
    **kwargs
):
    """
    Train the model for one gradient update.

    Args:
        model (torch.nn.Module): An instance of a Graph Neural Network (GNN).
        x (torch.flaot): Node feature matrix with shape [num_nodes, num_node_features].
        edge_index (torch.long): Edges in COO format with shape [2, num_edges] for
            edges in the training set.
        y (torch.long): Ground truth label for the training set.
        mask (torch.bool): Mask for GNN output in the training set.
        optimizer (torch.optim.Optimizer): Training optimizer.
        sig_coeffs (dict or None): Coefficients for SIG regularization loss. None for
            non-SIG models.
    
    Return:
        No object returned. Model parameters are updated.
    """
    model.train()
    optimizer.zero_grad()
    if sig_coeffs is None:
        out = model(
            x, 
            edge_index, 
            **kwargs
        )
        loss = class_loss.cross_entropy_loss(out[mask], y)
    else:
        out, x_prob, edge_prob = model(
            x, 
            edge_index, 
            return_probs=True, 
            **kwargs
        )

        loss_x_l1, \
        loss_x_ent, \
        loss_edge_l1, \
        loss_edge_ent = reg_loss.reg_sig_loss(
            x_prob, 
            edge_prob, 
            sig_coeffs
        )

        loss = class_loss.cross_entropy_loss(out[mask], y) + \
            loss_x_l1 + loss_x_ent + loss_edge_l1 + loss_edge_ent
    loss.backward(retain_graph=True)
    optimizer.step()

def batch_train(
    model,
    loader,
    optimizer,
    device,
    sig_coeffs=None,
    **kwargs
):
    """
    Train the model for multiple gradient updates through batches of data.

    Args:
        model (torch.nn.Module): An instance of a Graph Neural Network (GNN).
        loader (torch_geometric.data.DataLoader): Data loader for the training set, 
            with data attributes x, edge_index, y.
        optimizer (torch.optim.Optimizer): Training optimizer.
        device (torch.device): Model device.
        sig_coeffs (dict or None): Coefficients for SIG regularization loss. None for
            non-SIG models.
    
    Return:
        No object returned. Model parameters are updated.
    """
    model.train()
    for data_train in loader:
        optimizer.zero_grad()
        data_train = data_train.to(device)
        if sig_coeffs is None:
            out = model(
                data_train.x, 
                data_train.edge_index,
                batch=data_train.batch,
                **kwargs
            )
            loss = class_loss.cross_entropy_loss(out, data_train.y.view(-1))
        else:
            out, x_prob, edge_prob = model(
                data_train.x, 
                data_train.edge_index, 
                return_probs=True, 
                batch=data_train.batch,
                **kwargs
            )
            loss_x_l1, \
            loss_x_ent, \
            loss_edge_l1, \
            loss_edge_ent = reg_loss.reg_sig_loss(
                x_prob, 
                edge_prob, 
                sig_coeffs
            )
            loss = class_loss.cross_entropy_loss(out, data_train.y.view(-1)) + \
                loss_x_l1 + loss_x_ent + loss_edge_l1 + loss_edge_ent
        loss.backward()
        optimizer.step()

def evaluate(
    model,
    x, 
    edge_index, 
    y,
    mask,
    sig_coeffs=None,
    **kwargs
):
    """
    Evaluate the performance of a model.

    Args:
        model (torch.nn.Module): An instance of a Graph Neural Network (GNN).
        x (torch.flaot): Node feature matrix with shape [num_nodes, num_node_features].
        edge_index (torch.long): Edges in COO format with shape [2, num_edges] for
            edges in the evaluation set.
        y (torch.long): Ground truth label for the evaluation set.
        mask (torch.bool): Mask for GNN output in the evaluation set.
        sig_coeffs (dict or None): Coefficients for SIG regularization loss. None for
            non-SIG models.

    Return:
        results (dict): Evaluation results with the following values
            accuracy (float): Accuracy of the model prediction.
            auroc (float): Area under the ROC curve of the model output.
            loss_class (float): Classification loss of the model output.
            loss_x_l1, loss_x_ent, loss_edge_l1, loss_edge_ent (float or None): 
                Node feature and edge probability regularization losses if sig_coeffs is 
                not None.
    """
    model.eval()
    if sig_coeffs is None:
        with torch.no_grad():
            out = model(
                x, 
                edge_index,
                **kwargs
            )
            loss_x_l1 = None
            loss_x_ent = None
            loss_edge_l1 = None
            loss_edge_ent = None
    else:
        with torch.no_grad():
            out, x_prob, edge_prob = model(
                x, 
                edge_index, 
                return_probs=True, 
                **kwargs
            )

        loss_x_l1, \
        loss_x_ent, \
        loss_edge_l1, \
        loss_edge_ent = reg_loss.reg_sig_loss(
            x_prob, 
            edge_prob, 
            sig_coeffs
        )
        loss_x_l1 = loss_x_l1.item()
        loss_x_ent = loss_x_ent.item()
        loss_edge_l1 = loss_edge_l1.item()
        loss_edge_ent = loss_edge_ent.item()
    
    out = out[mask]
    pred = out.max(1)[1]
    loss_class = class_loss.cross_entropy_loss(out, y)
    loss_class = loss_class.item()

    out = out.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    y_categories = [np.sort(np.unique(y))]
    onehot_encoder = OneHotEncoder(y_categories, sparse=False)
    y_onehot = onehot_encoder.fit_transform(y[:,np.newaxis])

    accuracy = metrics.accuracy_score(y, pred)
    auroc = metrics.roc_auc_score(
        y_onehot, 
        out, 
        average='macro'
    )
    results = {
        'accuracy': accuracy,
        'auroc': auroc,
        'loss_class': loss_class,
        'loss_x_l1': loss_x_l1,
        'loss_x_ent': loss_x_ent,
        'loss_edge_l1': loss_edge_l1,
        'loss_edge_ent': loss_edge_ent,
    }
    return results

def batch_evaluate(
    model,
    loader,
    device,
    sig_coeffs=None,
    **kwargs
):
    """
    Evaluate the performance of a model by running it through batches.

    Args:
        model (torch.nn.Module): An instance of a Graph Neural Network (GNN).
        loader (torch_geometric.data.DataLoader): Data loader for the evaluation set, 
            with data attributes x, edge_index, y.
        device (torch.device): Model device.
        sig_coeffs (dict or None): Coefficients for SIG regularization loss. None for
            non-SIG models.

    Return:
        results (dict): Evaluation results with the following values
            accuracy (float): Accuracy of the model prediction.
            auroc (float): Area under the ROC curve of the model output.
            loss_class (float): Classification loss of the model output.
            loss_x_l1, loss_x_ent, loss_edge_l1, loss_edge_ent (float or None): 
                Node feature and edge probability regularization losses if sig_coeffs is 
                not None.
    """
    model.eval()
    outs = []
    preds = []
    ys = []
    y_onehots = []
    total_num_loss_class = 0
    total_loss_class = 0

    if sig_coeffs is None:
        total_num_loss_x = None
        total_num_loss_edge = None
        total_loss_x_l1 = None
        total_loss_x_ent = None
        total_loss_edge_l1 = None
        total_loss_edge_ent = None
    else:
        total_num_loss_x = 0
        total_num_loss_edge = 0
        total_loss_x_l1 = 0
        total_loss_x_ent = 0
        total_loss_edge_l1 = 0
        total_loss_edge_ent = 0

    for data_eval in loader:
        data_eval = data_eval.to(device)
        if sig_coeffs is None:
            with torch.no_grad():
                out = model(
                    data_eval.x, 
                    data_eval.edge_index,
                    batch=data_eval.batch,
                    **kwargs
                )
        else:
            with torch.no_grad():
                out, x_prob, edge_prob = model(
                    data_eval.x, 
                    data_eval.edge_index, 
                    return_probs=True, 
                    batch=data_eval.batch,
                    **kwargs
                )

            loss_x_l1, \
            loss_x_ent, \
            loss_edge_l1, \
            loss_edge_ent = reg_loss.reg_sig_loss(
                x_prob, 
                edge_prob, 
                sig_coeffs
            )
            
            num_loss_x = x_prob.shape[0]
            num_loss_edge = edge_prob.shape[0]

            total_num_loss_x += num_loss_x
            total_num_loss_edge += num_loss_edge
            total_loss_x_l1 += loss_x_l1.item() * num_loss_x
            total_loss_x_ent += loss_x_ent.item() * num_loss_x
            total_loss_edge_l1 += loss_edge_l1.item() * num_loss_edge
            total_loss_edge_ent += loss_edge_ent.item() * num_loss_edge

        num_loss_class = out.shape[0]
        loss_class = class_loss.cross_entropy_loss(out, data_eval.y.view(-1))

        total_num_loss_class += num_loss_class
        total_loss_class += loss_class.item() * num_loss_class
        
        pred = out.max(1)[1]

        pred = pred.detach().cpu().numpy()
        out = out.detach().cpu().numpy()
        y = data_eval.y.detach().cpu().numpy()

        y_categories = [np.arange(model.output_size)]
        onehot_encoder = OneHotEncoder(y_categories, sparse=False)
        y_onehot = onehot_encoder.fit_transform(y[:, np.newaxis])

        outs.append(out)
        preds.append(pred)
        ys.append(y)
        y_onehots.append(y_onehot)
    outs = np.concatenate(outs)
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    y_onehots = np.concatenate(y_onehots)
    
    accuracy = metrics.accuracy_score(ys, preds)
    auroc = metrics.roc_auc_score(
        y_onehots, 
        outs, 
        average='macro'
    )
    results = {
        'accuracy': accuracy,
        'auroc': auroc,
        'loss_class': total_loss_class / total_num_loss_class,
        'loss_x_l1': total_loss_x_l1 / total_num_loss_x \
            if total_loss_x_l1 is not None else total_loss_x_l1,
        'loss_x_ent': total_loss_x_ent / total_num_loss_x \
            if total_loss_x_ent is not None else total_loss_x_ent,
        'loss_edge_l1': total_loss_edge_l1 / total_num_loss_edge \
            if total_loss_edge_l1 is not None else total_loss_edge_l1,
        'loss_edge_ent': total_loss_edge_ent / total_num_loss_edge \
            if total_loss_edge_ent is not None else total_loss_edge_ent,
    }
    return results

def init_model(model):
    """
    Initialize model parameters with Xavier uniform initialization.

    Args:
        model (torch.nn.Module): An instance of a Graph Neural Network (GNN).

    Return:
        No ojbect returned. Model parameters are intialized.
    """
    for name, param in model.named_parameters():
        with torch.no_grad():        
            if len(param.shape) == 1:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                else:
                    init.xavier_uniform_(
                        param.view(-1, 1), 
                        gain=init.calculate_gain('relu')
                    )
            else:
                init.xavier_uniform_(
                    param, 
                    gain=init.calculate_gain('relu')
                )

def init_adam(
    model, 
    lr, 
    weight_decay
):
    """
    Initialize an Adam optimizer.

    Args:
        model (torch.nn.Module): An instance of a Graph Neural Network (GNN), with 
            intialized paramters.
        lr (float): Learning rate.
        weight_decay (float): L2 penalty coefficient on all model parameters.
    
    Return:
        optimizer (torch.optim.Adam): An instance of Adam optimizer.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    return optimizer

def run_model(
    model,
    optimizer,
    num_epochs,
    kwargs_train,
    kwargs_eval,
    sig_coeffs=None,
    verbosity=0, 
    verbosity_period=100,
    name_model=0,
    name_train='train',
    name_eval='test',
    fn_train=train,
    fn_eval=evaluate,
    writer=None,
    log_file=None,
    **kwargs
):
    """
    Run an initialized model through epochs of training and evaluation.

    Args:
        model (torch.nn.Module): An instance of a Graph Neural Network (GNN).
        optimizer (torch.optim.Optimizer): Training optimizer.
        num_epochs(int): Number of epochs to run.
        kwargs_train (dict): kwargs for the training data.
        kwargs_eval (dict): kwargs for evaluation data.
        sig_coeffs (dict or None): Coefficients for SIG regularization loss. None for
            non-SIG models.
        verbosity (int): Level of verbosity {0, 1, 2}.
        verbosity_period (int): Period of epochs to log progress.
        name_model (int or str): Name of model to log.
        name_train (str): Name of training data to log.
        name_eval (str): Name of evaluation data to log.
        fn_train (function): Training function (train, batch_train).
        fn_eval (function): Evaluation function (evaluate, batch_evaluate).
        writer (torch.utils.tensorboard.SummaryWriter or None): Tensorboard writer.
        log_file (str or None): Log file name.
    
    Return:
        model (torch.nn.Module): The trained model.
        results_train (dict): Performance of the training data by the trained model, with
            keys 'accuracy', 'auroc', 'loss_class', 'loss_x_l1', 'loss_x_ent', 
            'loss_edge_l1', 'loss_edge_ent'.
        results_eval (dict): Performance of the evaluation data by the trained model, with
            the same keys as results_train.
    """
    assert verbosity in [0, 1, 2], \
        'verbosity is {}. verbosity has to be 0, 1, or 2'.format(verbosity)
    start = time.time()
    for i in range(num_epochs):
        fn_train(
            model=model,
            optimizer=optimizer, 
            sig_coeffs=sig_coeffs,
            **kwargs_train,
            **kwargs
        )
        results_train = fn_eval(
            model=model,
            sig_coeffs=sig_coeffs,
            **kwargs_eval,
            **kwargs
        )
        results_eval = fn_eval(
            model=model,
            sig_coeffs=sig_coeffs,
            **kwargs_eval,
            **kwargs
        )
        epoch = i + 1
        runtime = time.time() - start

        if verbosity == 2 and epoch % verbosity_period == 0:
            io_utils.log_model_run(
                results_train,
                results_eval,
                name_model, 
                name_train,
                name_eval,
                epoch,
                runtime,
                writer,
                log_file
            )
    if verbosity == 1:
        io_utils.log_model_run(
            results_train,
            results_eval,
            name_model, 
            name_train,
            name_eval,
            epoch,
            runtime,
            writer,
            log_file
        )
    return model, results_train, results_eval

def get_hyperparam_combos(hyperparams):
    """
    Get all the combinations of the hyperparameters.

    Args:
        hyperparams (dict): Dictionary with hyperparameter names as keys and lists of
            hyperparameter values as values.
    
    Return:
        hyperparam_combos (list of dict): Hyperparameter combination list. Each item is a
            dictionary with hyperparameter names as keys and hyperparameter values as
            values.
    """
    names = [name for name in hyperparams.keys()]
    combos = itertools.product(
        *[hyperparams[name] for name in names]
    )
    hyperparam_combos = []
    for combo in combos:
        hyperparam_combos.append(
            {names[i]: combo[i] for i in range(len(names))}
        )
    return hyperparam_combos

def get_combo_sig_coeffs(combo):
    """
    Get SIG coefficients from hyperparameter combination.

    Args:
        combo (dict): Hyperparameter combination, a
            dictionary with hyperparameter names as keys and hyperparameter values as
            values.
    
    Return:
        sig_coeffs (dict): SIG regularization coefficients with keys 
            {'x_l1', 'x_ent', 'edge_l1', 'edge_ent'}.
    """
    sig_coeffs = {
        'x_l1': combo['x_l1_coeff'],
        'x_ent': combo['x_ent_coeff'],
        'edge_l1': combo['edge_l1_coeff'],
        'edge_ent': combo['edge_ent_coeff']
    }
    sig_coeffs_none = np.array(
        [value is None for value in sig_coeffs.values()]
    )
    if np.sum(sig_coeffs_none) > 0:
        sig_coeffs = None
    return sig_coeffs


def run_combo_model(
    model_type,
    combo,
    input_size,
    output_size,
    device,
    classify_graph=False,
    **kwargs
):
    """
    Run a model based on a combination of hyperparameters.

    Args:
        model_type (str): Type of GNN {'GCN', 'SIGCN', 'GAT'}.
        combo (dict): Hyperparameter combination, a
            dictionary with hyperparameter names as keys and hyperparameter values as
            values.
        input_size (int): Input node feature dimension.
        output_size (int): GNN final output dimension.
        device (torch.device): Model device.
        classify_graph (boolean): Whether the model is a graph classifier. Default False
            for node classifier.
    
    Return:
        Output of run_model.
    """
    model = eval(model_type)(
        input_size=input_size,
        output_size=output_size,
        hidden_conv_sizes=(combo['hidden_dim'], ) * combo['num_hidden_layer'],
        hidden_dropout_probs=(combo['dropout_rate'], ) * combo['num_hidden_layer'],
        classify_graph=classify_graph,
        lin_dropout_prob=combo['dropout_rate'] if classify_graph else None
    ).to(device)
    init_model(model)
    optimizer = init_adam(
        model, 
        combo['learning_rate'], 
        combo['l2_coeff']
    )
    sig_coeffs = get_combo_sig_coeffs(combo)
    return run_model(
        model=model, 
        optimizer=optimizer,
        sig_coeffs=sig_coeffs,
        num_epochs=combo['num_epochs'], 
        **kwargs
    )

def tune_hyperparams(
    model_type,
    input_size, 
    output_size,
    num_hidden_layers,
    hidden_dims,
    learning_rates,
    nums_epochs,
    l2_coeffs,
    dropout_rates,
    device,
    x_l1_coeffs=[None],
    x_ent_coeffs=[None],
    edge_l1_coeffs=[None],
    edge_ent_coeffs=[None],
    criterion='auroc',
    classify_graph=False,
    **kwargs
):
    """
    Tune the hyperparameters of a Graph Neural Network (GNN) model, using a training 
        dataset and an evaluation dataset.

    Args:
        model_type (str): Type of GNN {'GCN', 'SIGCN', 'GAT'}.
        input_size (int): Input node feature dimension.
        output_size (int): GNN final output dimension.
        num_hidden_layers (list of int): Numbers of hidden layers to try.
        hidden_dims (list of int): Hidden dimension sizes to try.
        learning_rates (list of float): Learning rates to try.
        nums_epochs (list of int): Epoch numbers to try.
        l2_coeffs (list of float): All-parameter L2 penalty coefficients to try.
        dropout_rates (list of float): All-hidden layer dropout rates to try.
        device (torch.device): Model device.
        x_l1_coeffs (list of flaot or None): Node feature probability L1 penalty 
            coefficients to try (for SIG).
        x_ent_coeffs (list of float or None): Node feature probability entropy penalty 
            coefficients to try (for SIG).
        edge_l1_coeffs (list of float or None): Edge probability L1 penalty coefficients
            to try (for SIG).
        edge_ent_coeffs (list of float or None): Edge probability entropy penalty
            coefficients to try (for SIG).
        criterion (str): Tuning criterion, one of the keys in results_* from run_model
            output.
        classify_graph (boolean): Whether the model is a graph classifier. Default False
            for node classifier.

    Return:
        best_model (torch.nn.Module): Best GNN model.
        best_hyperparams (dict): Best hyperparameters based on the evaluation data and 
            tuning criterion, with keys {'num_hidden_layer', 'hidden_dim', 
            'learning_rate', 'num_epoch', 'l2_coeff', 'dropout_rate', 
            'x_l1_coeff', 'x_ent_coeff', 'edge_l1_coeff', 'edge_ent_coeff'}
        best_results_train (dict): Best performance of the training data by the model, 
            with keys 'accuracy', 'auroc', 'loss_class', 'loss_x_l1', 'loss_x_ent', 
            'loss_edge_l1', 'loss_edge_ent'.
        best_results_eval (dict): Best performance of the evaluation data by the model,
            with the same keys as best_results_train.
    """
    hyperparams = {
        'num_hidden_layer': num_hidden_layers,
        'hidden_dim': hidden_dims, 
        'learning_rate': learning_rates,
        'num_epochs': nums_epochs,
        'l2_coeff': l2_coeffs,
        'dropout_rate': dropout_rates,
        'x_l1_coeff': x_l1_coeffs,
        'x_ent_coeff': x_ent_coeffs,
        'edge_l1_coeff': edge_l1_coeffs,
        'edge_ent_coeff': edge_ent_coeffs
    }
    hyperparam_combos = get_hyperparam_combos(hyperparams)

    best_model = None
    best_hyperparams = None
    best_results_train = {criterion: -1}
    best_results_eval = {criterion: -1}

    for i in range(len(hyperparam_combos)):
        combo = hyperparam_combos[i]
        model, \
        results_train, \
        results_eval = run_combo_model(
            model_type=model_type,
            combo=combo,
            input_size=input_size,
            output_size=output_size,
            device=device,
            name_model=i, 
            classify_graph=classify_graph,
            **kwargs
        )
        if results_eval[criterion] > best_results_eval[criterion]:
            best_model = model
            best_hyperparams = combo
            best_results_train = results_train
            best_results_eval = results_eval
    return best_model, best_hyperparams, best_results_train, best_results_eval

def evaluate_explanation(
    explainer,
    node_indices,
    x,
    edge_index,
    min_subgraph_size,
    edge_imp=None,
    node_feat_imp=None,
    overall_node_feat_imp=None,
    verbose=True,
    verbosity_period=10,
    log_file=None,
    log_prefix='',
    **kwargs
):
    """
    Evaluate an explainer's explanation performance.
    
    Args:
        explainer (torch.nn.Module): Explainer with explain_node function.
        node_indices (list): Nodes to explain.
        x (torch.float): Node feature matrix for all the nodes with shape 
            [num_nodes, num_node_feats].
        edge_index (torch.long): Edge COO for all the edges with shape [2, num_edges].
        min_subgraph_size (int): Minimum important subgraph size for accuracy evaluation.
        edge_imp (torch.long or None): Ground truth edge importance label, with shape
            [num_nodes, num_edges].
        node_feat_imp (torch.long or None): Node-specific ground truth node feature 
            importance label with shape [num_nodes, num_node_feats].
        overall_node_feat_imp (torch.long or None): Ground truth node feature importance
            label applicable to all the nodes, with shape [num_node_feats].
        verbose (boolean): Whether to print and log progress.
        verbosity_period (int): Number of nodes to go through for one progress message.
        log_file (str or None): File path for logging progress.

    Return:
        results (dict): Result names and values.
    """
    top_node_feat_sizes = []
    all_node_feat_scores, all_node_feat_preds, all_node_feat_imps = [], [], []

    # Computational subgraphs
    comp_sizes = []
    comp_edge_scores, comp_edge_preds, comp_edge_imps = [], [], []

    # Extracted important subgraphs
    ext_sizes, ext_thresholds = [], []

    counter = 0
    start = time.time()
    for node_index in node_indices:
        if isinstance(explainer, GNNExplainer):
            comp_x, _, comp_edge_mask, _ = explainer.__subgraph__(
                node_index,
                x,
                edge_index
            )
        else:
            comp_x, _, _, comp_edge_mask = explainer.__subgraph__(
                node_index,
                x,
                edge_index
            )
        comp_sizes.append(comp_x.shape[0])

        node_feat_score, edge_score = explainer.explain_node(
            node_index, 
            x,
            edge_index,
            **kwargs
        )

        # minmax normalization for the node feature score
        if node_feat_score is not None:
            node_feat_score -= torch.min(node_feat_score)
            if torch.max(node_feat_score) != 0:
                node_feat_score /= torch.max(node_feat_score)

        if node_feat_score is not None and node_feat_imp is not None:
            node_index_feat_imp = node_feat_imp[node_index, :]
            all_node_feat_scores.append(node_feat_score)
            all_node_feat_imps.append(node_index_feat_imp)

            top_k = torch.sum(node_index_feat_imp).item()
            top_node_feat_index = torch.topk(node_feat_score, top_k).indices
            all_node_feat_pred = torch.zeros(
                node_index_feat_imp.shape[0]
            ).type(torch.LongTensor)
            all_node_feat_pred[top_node_feat_index] = 1
            all_node_feat_preds.append(all_node_feat_pred)
            top_node_feat_sizes.append(top_k)
        
        if edge_score is not None and edge_imp is not None:
            comp_edge_index = edge_index[:, comp_edge_mask]
            comp_edge_score = edge_score[comp_edge_mask].detach()
            comp_edge_imp = edge_imp[node_index, comp_edge_mask]

            comp_edge_scores.append(comp_edge_score)
            comp_edge_imps.append(comp_edge_imp)
            
            ext_edge_mask, \
            ext_node_index, \
            ext_size, \
            ext_threshold = graph_utils.extract_important_subgraph(
                [node_index],
                comp_edge_index,
                comp_edge_score,
                min_subgraph_size
            )
            comp_edge_preds.append(ext_edge_mask.type(torch.LongTensor))
            ext_sizes.append(ext_size)
            ext_thresholds.append(ext_threshold)

        runtime = time.time() - start
        counter += 1
        if counter % verbosity_period == 0:
            mesg = 'finished computing explanations for {}/{} nodes'.format(
                counter,
                len(node_indices)
            )
            io_utils.print_log(
                mesg, 
                prefix=log_prefix,
                runtime=runtime,
                log_file=log_file
            )

    overall_node_feat_prop = None
    overall_node_feat_accuracy, \
    overall_node_feat_auroc, \
    overall_node_feat_precision, \
    overall_node_feat_recall, \
    overall_node_feat_f1 = None, None, None, None, None
    overall_top_node_feat_size = None

    all_node_feat_prop = None
    all_node_feat_accuracy, \
    all_node_feat_auroc, \
    all_node_feat_precision, \
    all_node_feat_recall, \
    all_node_feat_f1 = None, None, None, None, None
    if len(all_node_feat_imps) > 0:
        if overall_node_feat_imp is not None:
            overall_top_node_feat_size = torch.sum(overall_node_feat_imp).item()
            overall_node_feat_scores = torch.mean(
                torch.stack(all_node_feat_scores),
                dim=0
            )
            overall_top_node_feat_index = torch.topk(
                overall_node_feat_scores,
                overall_top_node_feat_size
            ).indices

            overall_node_feat_preds = torch.zeros(
                overall_node_feat_imp.shape[0]
            ).type(torch.LongTensor)
            overall_node_feat_preds[overall_top_node_feat_index] = 1

            overall_node_feat_scores = overall_node_feat_scores.detach().cpu().numpy()
            overall_node_feat_preds = overall_node_feat_preds.detach().cpu().numpy()
            overall_node_feat_imp = overall_node_feat_imp.detach().cpu().numpy()

            overall_node_feat_prop = np.mean(overall_node_feat_imp)
            overall_node_feat_accuracy = metrics.accuracy_score(
                overall_node_feat_imp, overall_node_feat_preds
            )
            overall_node_feat_auroc = metrics.roc_auc_score(
                overall_node_feat_imp, overall_node_feat_scores
            )
            overall_node_feat_precision = metrics.precision_score(
                overall_node_feat_imp, overall_node_feat_preds
            )
            overall_node_feat_recall = metrics.recall_score(
                overall_node_feat_imp, overall_node_feat_preds
            )
            overall_node_feat_f1 = metrics.f1_score(
                overall_node_feat_imp, overall_node_feat_preds
            )
        all_node_feat_scores = torch.cat(all_node_feat_scores).detach().cpu().numpy()
        all_node_feat_preds = torch.cat(all_node_feat_preds).detach().cpu().numpy()
        all_node_feat_imps = torch.cat(all_node_feat_imps).detach().cpu().numpy()

        all_node_feat_prop = np.mean(all_node_feat_imps)
        all_node_feat_accuracy = metrics.accuracy_score(
            all_node_feat_imps, all_node_feat_preds
        )
        all_node_feat_auroc = metrics.roc_auc_score(
            all_node_feat_imps, all_node_feat_scores
        )
        all_node_feat_precision = metrics.precision_score(
            all_node_feat_imps, all_node_feat_preds
        )
        all_node_feat_recall = metrics.recall_score(
            all_node_feat_imps, all_node_feat_preds
        )
        all_node_feat_f1 = metrics.f1_score(
            all_node_feat_imps, all_node_feat_preds
        )
    
    comp_edge_prop = None
    ext_edge_accuracy, \
    ext_edge_precision, \
    ext_edge_recall, \
    ext_edge_f1 = None, None, None, None
    if len(comp_edge_imps) > 0:
        comp_edge_imps = torch.cat(comp_edge_imps).detach().cpu().numpy()
        comp_edge_preds = torch.cat(comp_edge_preds).detach().cpu().numpy()
        comp_edge_scores = torch.cat(comp_edge_scores).detach().cpu().numpy()

        comp_edge_prop = np.mean(comp_edge_imps)
        ext_edge_accuracy = metrics.accuracy_score(comp_edge_imps, comp_edge_preds)
        ext_edge_precision = metrics.precision_score(comp_edge_imps, comp_edge_preds)
        ext_edge_recall = metrics.recall_score(comp_edge_imps, comp_edge_preds)
        ext_edge_f1 = metrics.f1_score(comp_edge_imps, comp_edge_preds)
    
    top_node_feat_sizes = np.array(top_node_feat_sizes)
    comp_sizes = np.array(comp_sizes)
    ext_sizes = np.array(ext_sizes)
    ext_thresholds = np.array(ext_thresholds)

    results = {
        'overall_node_feat_prop': overall_node_feat_prop,
        'overall_node_feat_accuracy': overall_node_feat_accuracy,
        'overall_node_feat_auroc': overall_node_feat_auroc,
        'overall_node_feat_precision': overall_node_feat_precision,
        'overall_node_feat_recall': overall_node_feat_recall,
        'overall_node_feat_f1': overall_node_feat_f1,
        'overall_top_node_feat_size': overall_top_node_feat_size,

        'all_node_feat_prop': all_node_feat_prop,
        'all_node_feat_accuracy': all_node_feat_accuracy,
        'all_node_feat_auroc': all_node_feat_auroc,
        'all_node_feat_precision': all_node_feat_precision,
        'all_node_feat_recall': all_node_feat_recall,
        'all_node_feat_f1': all_node_feat_f1,
        'top_node_feat_size_mean': np.mean(top_node_feat_sizes),
        'top_node_feat_size_std': np.std(top_node_feat_sizes),
        
        'comp_edge_prop': comp_edge_prop,
        'comp_size_mean': np.mean(comp_sizes),
        'comp_size_std': np.std(comp_sizes),

        'ext_edge_accuracy': ext_edge_accuracy,
        'ext_edge_precision': ext_edge_precision,
        'ext_edge_recall': ext_edge_recall,
        'ext_edge_f1': ext_edge_f1,
        'ext_size_mean': np.mean(ext_sizes),
        'ext_size_std': np.std(ext_sizes),
        'ext_threshold_mean': np.mean(ext_thresholds),
        'ext_threshold_std': np.std(ext_thresholds)
    }
    return results