#!/bin/bash 

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
gpu=           # will be deprecated, please use ngpu
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32          # numebr of parallel jobs
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
fs=16000         # sampling frequency
fmax=""          # maximum frequency
fmin=""          # minimum frequency
n_mels=80        # number of mel basis
# ESPNnet config
n_fft=512        # number of fft points
n_shift=160      # number of shift points
win_length=400   # number of samples in analysis window
# Tacotron config
taco_n_fft=1024      # number of fft points
taco_n_shift=512     # number of shift points
taco_win_length=1024 # number of samples in analysis window
#
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=blstmp     # encoder architecture type
elayers=4
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
spk_embed_dim=512
dlayers=1
dunits=300
# attention related
atype=location
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=20

# tacotron loss
tacotron_model=../tts1/exp/train_no_dev_taco2_enc512-3x5x512-1x512_dec2x1024_pre2x256_post5x5x512_att128-15x32_cm_bn_cc_msk_pw20.0_do0.5_zo0.1_lr1e-3_ep1e-6_wd0.0_bs64_sd1/results/model.conf

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.5
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# data
datadir=./downloads
an4_root=${datadir}/an4
data_url=http://www.speech.cs.cmu.edu/databases/an4/

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# check gpu option usage
if [ ! -z $gpu ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi

# Check trained tacotron model exists
if [ ! -f ${tacotron_model} ];then
    echo "Missing trained tacotron model in ../tts1/!\n\n${tacotron_model}\n"    
    exit
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodev
train_dev=train_dev
recog_set="train_dev test"

if [ ${stage} -le -1 ]; then
    echo "stage -1: Data Download"
    mkdir -p ${datadir}
    local/download_and_untar.sh ${datadir} ${data_url}
fi

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    mkdir -p data/{train,test} exp

    if [ ! -f ${an4_root}/README ]; then
        echo Cannot find an4 root! Exiting...
        exit 1
    fi

    python local/data_prep.py ${an4_root} ${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe

    for x in test train; do
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in test train; do
        # Using librosa
        local/make_fbank.sh --cmd "${train_cmd}" --nj 8 \
            --fs ${fs} --fmax "${fmax}" --fmin "${fmin}" \
            --n_mels ${n_mels} --n_fft ${n_fft} \
            --n_shift ${n_shift} --win_length $win_length \
            data/${x} exp/make_fbank/${x} ${fbankdir}
    done

    # make a dev set
    utils/subset_data_dir.sh --first data/train 100 data/${train_dev}
    n=$[`cat data/train/text | wc -l` - 100]
    utils/subset_data_dir.sh --last data/train ${n} data/${train_set}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

taco_feat_tr_dir=${dumpdir}/taco_${train_set}/delta${do_delta};mkdir -p ${taco_feat_tr_dir}
taco_feat_dt_dir=${dumpdir}/taco_${train_dev}/delta${do_delta};mkdir -p ${taco_feat_dt_dir}
if [ ${stage} -le 3 ]; then
    echo "stage 3: Tacotron Feature Generation"
    fbankdir=taco_fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch
    # on each frame
    for x in test train; do
        utils/copy_data_dir.sh data/${x} data/taco_${x}
        # Using librosa
        local/make_fbank.sh --cmd "${train_cmd}" --nj 8 \
            --fs ${fs} --fmax "${fmax}" --fmin "${fmin}" \
            --n_mels ${n_mels} --n_fft ${taco_n_fft} \
            --n_shift ${taco_n_shift} --win_length $taco_win_length \
            data/taco_${x} exp/taco_make_fbank/${x} ${fbankdir}
    done

    # make a dev set
    utils/subset_data_dir.sh --first data/taco_train 100 data/taco_${train_dev}
    n=$[`cat data/taco_train/text | wc -l` - 100]
    utils/subset_data_dir.sh --last data/taco_train ${n} data/taco_${train_set}

    # compute global CMVN
    compute-cmvn-stats scp:data/taco_${train_set}/feats.scp \
            data/taco_${train_set}/cmvn.ark

    # Dump features
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
        data/taco_${train_set}/feats.scp \
        data/taco_${train_set}/cmvn.ark exp/taco_dump_feats/train ${taco_feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
        data/taco_${train_dev}/feats.scp \
        data/taco_${train_set}/cmvn.ark exp/taco_dump_feats/dev ${taco_feat_dt_dir}
    for rtask in ${recog_set}; do
        # FIXME: Need to compose the path dynamically here, error prone    
        feat_recog_dir=${dumpdir}/taco_${rtask}/delta${do_delta}
        mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
            data/taco_${rtask}/feats.scp \
            data/taco_${train_set}/cmvn.ark exp/dump_feats/recog/taco_${rtask} \
            ${feat_recog_dir}
    done

    # Append data to jsons
    # feats.scp  spk2utt  text  utt2spk  wav.scp

    # Update json
    python local/data_io.py \
        --in-scp-file data/taco_${train_set}/feats.scp \
        --ark-class matrix \
        --input-name input2 \
        --in-json-file ${feat_tr_dir}/data.json \
        --action add-scp-data-to-input \
        --verbose 1
    python local/data_io.py \
        --in-scp-file data/taco_${train_dev}/feats.scp \
        --ark-class matrix \
        --input-name input2 \
        --in-json-file ${feat_dt_dir}/data.json \
        --action add-scp-data-to-input \
        --verbose 1
    python local/data_io.py \
        --in-scp-file data/taco_test/feats.scp \
        --ark-class matrix \
        --input-name input2 \
        --in-json-file ${dumpdir}/test/delta${do_delta}/data.json \
        --action add-scp-data-to-input \
        --verbose 1

fi

if [ ${stage} -le 4 ]; then
    echo "stage 4: x-vector extraction"

    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    for name in test train; do
        utils/copy_data_dir.sh data/${name} data/${name}_mfcc
        steps/make_mfcc.sh \
            --write-utt2num-frames true \
            --mfcc-config conf/mfcc.conf \
            --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc exp/make_mfcc $mfccdir
        utils/fix_data_dir.sh data/${name}_mfcc
        # TODO: I had to change this to 10
        sid/compute_vad_decision.sh --nj 10 --cmd "$train_cmd" \
            data/${name}_mfcc exp/make_vad ${vaddir}
        utils/fix_data_dir.sh data/${name}_mfcc
    done

    # Check pretrained model existence
    nnet_dir=exp/xvector_nnet_1a
    if [ ! -e $nnet_dir ];then
        echo "X-vector model does not exist. Download pre-trained model."
        wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
        tar xvf 0008_sitw_v2_1a.tar.gz
        mv 0008_sitw_v2_1a/exp/xvector_nnet_1a exp
        rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    fi
    # Extract x-vector
    for name in test train; do
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 10 \
            $nnet_dir data/${name}_mfcc \
            $nnet_dir/xvectors_${name}
    done

    # Append data to jsons
    # feats.scp  spk2utt  text  utt2spk  wav.scp

    # make a dev set from train
    cp data/train/{spk2utt,utt2spk,wav.scp} ${nnet_dir}/xvectors_train/ 
    cp ${nnet_dir}/xvectors_train/xvector.scp ${nnet_dir}/xvectors_train/feats.scp
    utils/subset_data_dir.sh --first $nnet_dir/xvectors_train 100 ${nnet_dir}/xvectors_${train_dev}
    n=$[`cat data/train/text | wc -l` - 100]
    utils/subset_data_dir.sh --last $nnet_dir/xvectors_train ${n} ${nnet_dir}/xvectors_${train_set}
    # Test
    cp ${nnet_dir}/xvectors_test/xvector.scp ${nnet_dir}/xvectors_test/feats.scp

    # Update json
    python local/data_io.py \
        --action add-scp-data-to-input \
        --in-scp-file ${nnet_dir}/xvectors_${train_set}/feats.scp \
        --ark-class vector \
        --input-name input3 \
        --in-json-file ${dumpdir}/${train_set}/delta${do_delta}/data.json \
        --verbose 1

    python local/data_io.py \
        --in-scp-file ${nnet_dir}/xvectors_${train_dev}/feats.scp \
        --ark-class vector \
        --input-name input3 \
        --in-json-file ${dumpdir}/${train_dev}/delta${do_delta}/data.json \
        --action add-scp-data-to-input \
        --verbose 1

    python local/data_io.py \
        --action add-scp-data-to-input \
        --in-scp-file ${nnet_dir}/xvectors_test/feats.scp \
        --ark-class vector \
        --input-name input3 \
        --in-json-file ${dumpdir}/test/delta${do_delta}/data.json \
        --verbose 1

fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 5 ]; then
    echo "stage 5: Network Training"
    #${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs} \
        --tts-model ${tacotron_model} \
        --expected-loss tts \
        --n-samples-per-input 2
fi

if [ ${stage} -le 6 ]; then
    echo "stage 8: Decoding"
    nj=8

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json 

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --verbose ${verbose} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            &
        wait

        score_sclite.sh ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi
