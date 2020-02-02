#!/usr/bin/env python3

def log_graph(writer, model, x, verbose=False):

    writer.add_graph(model, x, verbose=False)

def log_hparams(writer, args, final_test_loss, final_test_acc):

    options = vars(args)
    writer.add_hparams(hparam_dict=options, metric_dict={'acc': final_test_acc * 100.0, 'loss': final_test_loss * 100.0})