import torch
import numpy as np
import math
from torch.autograd import Variable
from e2e_asr_attctc_th import torch_is_old
from e2e_asr_attctc_th import to_cuda


def pad_ndarray_list(batch, pad_value):
    """FUNCTION TO PERFORM PADDING OF NDARRAY LIST

    :param list batch: list of the ndarray [(T_1, D), (T_2, D), ..., (T_B, D)]
    :param float pad_value: value to pad
    :return: padded batch with the shape (B, Tmax, D)
    :rtype: ndarray
    """
    bs = len(batch)
    maxlen = max([b.shape[0] for b in batch])
    if len(batch[0].shape) >= 2:
        batch_pad = np.zeros((bs, maxlen) + batch[0].shape[1:])
    else:
        batch_pad = np.zeros((bs, maxlen))
    batch_pad.fill(pad_value)
    for i, b in enumerate(batch):
        batch_pad[i, :b.shape[0]] = b

    return batch_pad


def get_tts_data(data, sort_by, parent, num_gpu, use_speaker_embedding=True, return_targets=True):
    # get eos
    eos = str(int(data[0][1]['output'][0]['shape'][1]) - 1)

    # get target features and input character sequence
    texts = [b[1]['output'][0]['tokenid'].split() + [eos] for b in data]
    feats = [b[1]['feat'] for b in data]

    # remove empty sequence and get sort along with length
    filtered_idx = filter(lambda i: len(texts[i]) > 0, range(len(feats)))
    if sort_by == 'feat':
        sorted_idx = sorted(filtered_idx, key=lambda i: -len(feats[i]))
    elif sort_by == 'text':
        sorted_idx = sorted(filtered_idx, key=lambda i: -len(texts[i]))
    else:
        logging.error("Error: specify 'text' or 'feat' to sort")
        sys.exit()
    texts = [np.fromiter(map(int, texts[i]), dtype=np.int64) for i in sorted_idx]
    feats = [feats[i] for i in sorted_idx]

    # get list of lengths (must be tensor for DataParallel)
    textlens = torch.from_numpy(np.fromiter((x.shape[0] for x in texts), dtype=np.int64))
    featlens = torch.from_numpy(np.fromiter((y.shape[0] for y in feats), dtype=np.int64))

    # perform padding and convert to tensor
    texts = torch.from_numpy(pad_ndarray_list(texts, 0)).long()
    feats = torch.from_numpy(pad_ndarray_list(feats, 0)).float()

    # make labels for stop prediction
    labels = feats.new(feats.size(0), feats.size(1)).zero_()
    for i, l in enumerate(featlens):
        labels[i, l - 1:] = 1

    if False: #torch_is_old:
        texts = to_cuda(parent, texts, volatile=not parent.training)
        feats = to_cuda(parent, feats, volatile=not parent.training)
        labels = to_cuda(parent, labels, volatile=not parent.training)
    else:
        texts = to_cuda(parent, texts)
        feats = to_cuda(parent, feats)
        labels = to_cuda(parent, labels)

    # load speaker embedding
    if use_speaker_embedding:
        # HERE: Read vectors when
        spembs = [da[1]['input'][2]['feat'] for da in data]
        spembs = [spembs[i] for i in sorted_idx]
        spembs = torch.from_numpy(np.array(spembs)).float()
        if torch_is_old:
            spembs = Variable(spembs, volatile=not parent.training)
        if num_gpu >= 0:
            spembs = spembs.cuda()
    else:
        spembs = None

    if return_targets:
        return texts, textlens, feats, labels, featlens, spembs
    else:
        return texts, textlens, feats, spembs


def get_chunk_loss(x, generate, n_samples_per_input, maxlenratio, minlenratio,sample_scaling, num_gpu, mtlalpha, model, parent):

    # Get samples for this batch subset 
    loss_ctc, loss_att, ys = generate(
        x,
        n_samples_per_input=n_samples_per_input,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio
    )

    # Expand the batch for each sample
    expanded_x = []
    import copy
    for example_index, example_x in enumerate(x):
        for n in range(n_samples_per_input): 

            # Assumes samples are placed consecutively
            text_sample = ys[example_index*n_samples_per_input + n]
            new_example_x = copy.deepcopy(example_x)

            # Remove ASR features
            del new_example_x[1]['input'][0]

            # Replace output sequence
            new_example_x[1]['output'][0]['shape'][1] = len(text_sample)
            new_example_x[1]['output'][0]['text'] = None
            new_example_x[1]['output'][0]['token'] = None
            new_example_x[1]['output'][0]['tokenid'] = " ".join(
                map(str, list(text_sample.data.cpu().numpy()))
            )
            expanded_x.append(new_example_x)

    # Number of gpus
    if num_gpu == 1:
        gpu_id = range(num_gpu)
    elif num_gpu > 1:
        gpu_id = range(num_gpu)
    else:
        gpu_id = [-1]

    # Construct a Tacotron batch from ESPNet batch and the samples
    # texts, textlens, feats, labels, featlens, spembs
    #taco_sample = get_tts_data(expanded_x, 'text', model, num_gpu, return_targets=True)
    # Tacotron converter
    from tts_pytorch import CustomConverter
    taco_converter = CustomConverter(
        gpu_id,
        use_speaker_embedding=True
    )
    samples = taco_converter([expanded_x])
    print("")

    # Merge losses and get the data for logging
    acc = 0.
    loss = None
    alpha = mtlalpha
    
    alpha = 0 # Debug
    
    if alpha == 0:
        loss = loss_att
        loss_att_data = loss_att.data[0] if torch_is_old else float(loss_att)
        loss_ctc_data = None
    elif alpha == 1:
        loss = loss_ctc
        loss_att_data = None
        loss_ctc_data = loss_ctc.data[0] if torch_is_old else float(loss_ctc)
    else:
        loss = alpha * loss_ctc + (1 - alpha) * loss_att
        loss_att_data = loss_att.data[0] if torch_is_old else float(loss_att)
        loss_ctc_data = loss_ctc.data[0] if torch_is_old else float(loss_ctc)
    
    # Get posterior probabilities from loss. We need to normalize
    # within the samples
    prob = sample_scaling * -loss
    # FIXME: Underflow problem at the momment
    prob = prob.view(len(x), n_samples_per_input)
    prob = torch.nn.Softmax(dim=1)(prob)
    # (batch_size * n_samples_per_input)
    prob = prob.view(-1)
   
    # 
    sample_loss = (
        1. / num_gpu * 
        model.loss_fn(*samples).mean(2).mean(1) * 
        prob
    ).mean()

    if math.isnan(sample_loss.data):
        import ipdb;ipdb.set_trace(context=50)
        print("")

    return sample_loss
