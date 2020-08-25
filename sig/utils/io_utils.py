# Utilities for IO


def get_runtime(sec):
    """
    Get the runtime in hour:minute:second.

    Args:
        sec (float): Runtime in seconds.
    
    Return:
        str for the formatted runtime.
    """
    hour = sec // 3600
    sec %= 3600
    minute = sec // 60
    sec %= 60
    return "%d:%02d:%02d" % (hour, minute, sec)

def print_log(
    mesg, 
    prefix='',
    runtime=None,
    log_file=None,
    mode='a'
):
    """
    Print and/or log a message.

    Args:
        mesg (str): Message to print log.
        prefix (str): Prefix for the message.
        runtime (float or None): Runtime in seconds to prefix the message.
        log_file (str or None): Log file name.
        mode (str): File opening mode for log_file.

    Return:
        No object returned. A message is printed and/or logged.
    """
    if runtime is not None:
        mesg = get_runtime(runtime) + ' ' + mesg
    mesg = prefix + mesg
    print(mesg)
    if log_file is not None:
        with open(log_file, mode) as f:
            f.write(mesg + '\n')

def get_result_mesg(results):
    """
    Get messages for model evaluation results.

    Args:
        results (dict): Evaluation results with None of float values.

    Return:
        mesgs (dict): Messages for the corresponding results.
    """
    mesgs = {}
    for key, val in results.items():
        mesg = key + ': '
        if val is None:
            mesg += 'None'
        else:
            mesg += '%0.4f' % val
        mesgs[key] = mesg
    return mesgs

def write_result_tb(
    writer,
    epoch,
    results,
    subtag=''
):
    """
    Write model evaluation results to tensorboard.

    Args:
        writer (torch.utils.tensorboard.SummaryWriter): Tensorboard writer.
        epoch (int): Epoch number.
        results (dict): Evaluation results with None of float values.
        subtag (str): Tag suffix.

    Return:
        No objects returned. Tensorboard log is updated.
    """
    for key, val in results.items():
        tag = key + '_' + subtag
        if val is not None:
            writer.add_scalar(tag, val, epoch)

def log_model_run(
    results_train,
    results_eval,
    name_model,
    name_train,
    name_eval,
    epoch,
    runtime,
    writer=None,
    log_file=None
):
    """
    Log the training and evaluation of a model at the end of an epoch.

    Args:
        results_train (dict): Model performance results for the training set.
        results_eval (dict): Model performance results for the evaluation set.
        name_model (int or str): Name of model to log.
        name_train (str): Name of training data to log.
        name_eval (str): Name of evaluation data to log.
        epoch (int): Epoch number.
        runtime (float): Epoch runtime in seconds.
        writer (torch.utils.tensorboard.SummaryWriter or None): Tensorboard writer.
        log_file (str or None): Log file name.
    """
    mesg_train = get_result_mesg(results_train)
    mesg_eval = get_result_mesg(results_eval)
    print_log(
        'Model {} epoch {}. Finished training and evaluation.'.format(
            name_model, epoch
        ),
        runtime=runtime,
        log_file=log_file
    )
    for mesg in mesg_train.values():
        print_log(
            mesg,
            prefix='\t' + name_train + ' ',
            log_file=log_file
        )
    print_log('', log_file=log_file)
    for mesg in mesg_eval.values():
        print_log(
            mesg,
            prefix='\t' + name_eval + ' ',
            log_file=log_file
        )
    print_log('', log_file=log_file)
    if writer is not None:
        write_result_tb(
            writer,
            epoch,
            results_train,
            subtag=name_train
        )
        write_result_tb(
            writer,
            epoch,
            results_eval,
            subtag=name_eval
        )