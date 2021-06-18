import torch, pickle, configparser, sys
from transformerfmodel import TransformerFModel
from train_fmodel import eprint


def main(config):
    f_config = config['FModel']
    #use_gpu = f_config.getboolean('UseGPU')
    pytorch_fn = f_config.get('TorchFilename')
    extra_params_fn = f_config.get('ExtraParamsFilename')
    f_config['UseGPU'] = 'no'

    init_params = pickle.load(open(extra_params_fn, 'rb'))

    model = TransformerFModel(
                f_config=f_config,
                catb_vocab_size=init_params['catb_vocab_size'],
                hvb_vocab_size=init_params['hvb_vocab_size'],
                hvf_vocab_size=init_params['hvf_vocab_size'],
                hva_vocab_size=init_params['hva_vocab_size'],
                output_dim=init_params['output_dim']
    )

#    if use_gpu:
#        model = model.cuda()

    model.eval()
    model.load_state_dict(torch.load(pytorch_fn))

    # S x N x E
    seq_length = 5
#    dummy_attn_input = torch.ones(seq_length, 1, 
#        2*f_config.getint('SemDim') + f_config.getint('SynDim') + model.max_depth)
#    dummy_coref_emb = torch.ones(seq_length, 1, f_config.getint('AntDim') + 1)
    # the ith dummy vector is filled with 0.1 * (i+1)
    # so  [0.1 0.1 0.1 ...], [0.2 0.2 0.2 ...], ...
    dummy_attn_input = torch.FloatTensor(seq_length, 1, 
        2*f_config.getint('SemDim') + f_config.getint('SynDim') + model.max_depth)
    dummy_coref_emb = torch.FloatTensor(seq_length, 1, f_config.getint('AntDim') + 1)
    for i in range(seq_length):
        for j in range(len(dummy_attn_input[i, 0])):
            dummy_attn_input[i, 0, j] = 0.1 * (i+1)
        for j in range(len(dummy_coref_emb[i, 0])):
            dummy_coref_emb[i, 0, j] = 0.1 * (i+1)
    dummy_pred = model.compute(dummy_attn_input, dummy_coref_emb, verbose=True)
    # only look at the last word
    fdec_log_scores = dummy_pred[-1, 0]
    fdec_scores = torch.exp(fdec_log_scores)
    fdec_scores_normalized = fdec_scores/sum(fdec_scores)
    print('Dummy output:')
    for fdec_prob in fdec_scores_normalized:
        print(fdec_prob.item())

    
if __name__ == '__main__':
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)

