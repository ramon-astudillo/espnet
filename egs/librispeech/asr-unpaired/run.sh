#!/bin/bash 

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
gpu=            # will be deprecated, please use ngpu
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false # true when using CNN

# Tacotron architecture
tacotron_model_conf=../tts1/exp/train_clean_100_taco2_enc512-3x5x512-1x512_dec2x1024_pre2x256_post5x5x512_att128-15x32_cm_bn_cc_msk_pw1.0_do0.5_zo0.1_lr1e-3_ep1e-6_wd0.0_bs32_sort_by_output_mli150_mlo400_sd1/results/model.conf
# Tacotron config
fs=16000    # sampling frequency
fmax=""     # maximum frequency
fmin=""     # minimum frequency
n_mels=80   # number of mel basis
taco_n_fft=1024      # number of fft points
taco_n_shift=512     # number of shift points
taco_win_length=1024 # number of samples in analysis window

# network archtecture
prior_model=../asr1/exp/train_100_blstmp_e8_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_bs50_mli800_mlo150/results/model.acc.best
# encoder related
etype=blstmp     # encoder architecture type
elayers=8
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=50
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# rnnlm related
lm_weight=0.3

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/12

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

# Check trained tts model exists
if [ ! -f ${tacotron_model} ];then
    printf "Missing trained tts model in\n\n${tacotron_model}\n\n"
    exit
fi

# Check trained asr model exists
if [ ! -f ${prior_model} ];then
    printf "Missing trained asr model in\n\n${prior_model}\n\n"
    exit
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# FIXME: These two should be literal set names e.g. train_clean_360
train_set=unpaired_360
train_dev=dev
recog_set="test_clean test_other dev_clean dev_other"

if [ ${stage} -le -1 ]; then
    echo "stage -1: Data Download"
    for part in dev-clean test-clean dev-other test-other train-clean-360; do
        local/download_and_untar.sh ${datadir} ${data_url} ${part}
    done
fi

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-360; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/$(echo ${part} | sed s/-/_/g)
    done

    # Uncoment this to debug using 1% of the data
    #mv data data.backup    
    #python local/subsample_data.py --in-data-folder data.backup/ --out-data-folder data/
    #for file in $(ls data);do bash utils/fix_data_dir.sh data/$file;done

fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in dev_clean test_clean dev_other test_other train_clean_360; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 data/${x} exp/make_fbank/${x} ${fbankdir}
    done

    utils/combine_data.sh data/${train_set}_org data/train_clean_360 
    utils/combine_data.sh data/${train_dev}_org data/dev_clean data/dev_other

    # remove utt having more than 1500 frames
    # remove utt having more than 300 characters
    remove_longshortdata.sh --maxframes 1500 --maxchars 300 data/${train_set}_org data/${train_set}
    remove_longshortdata.sh --maxframes 1500 --maxchars 300 data/${train_dev}_org data/${train_dev}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,12}/${USER}/${RANDOM}/espnet-data/egs/librispeech/asr-unpaired/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,12}/${USER}/${RANDOM}/espnet-data/egs/librispeech/asr-unpaired/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
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
    for x in ${train_set} ${train_dev} test_clean test_other dev_clean dev_other; do
        printf "make_fbank: \033[34m${x}\033[0m\n"    
        utils/copy_data_dir.sh data/${x} data/taco_${x}
        # Using librosa
        local/make_fbank.sh --cmd "${train_cmd}" --nj 40 \
            --fs ${fs} --fmax "${fmax}" --fmin "${fmin}" \
            --n_mels ${n_mels} --n_fft ${taco_n_fft} \
            --n_shift ${taco_n_shift} --win_length $taco_win_length \
            data/taco_${x} exp/taco_make_fbank/${x} ${fbankdir}
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/taco_${train_set}/feats.scp data/taco_${train_set}/cmvn.ark

    dump.sh --cmd "$train_cmd" --nj 80 --do_delta false \
        data/taco_${train_set}/feats.scp data/taco_${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta false \
        data/taco_${train_dev}/feats.scp data/taco_${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/taco_${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta false \
            data/taco_${rtask}/feats.scp data/taco_${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done

    # Append data to jsons

    # Update json
    python local/data_io.py \
        --in-scp-file data/taco_${train_set}/feats.scp \
        --ark-class matrix \
        --input-name input2 \
        --in-json-file ${feat_tr_dir}/data.json \
        --action add-scp-data-to-input \
        --verbose 1 \
        --force

    python local/data_io.py \
        --in-scp-file data/taco_${train_dev}/feats.scp \
        --ark-class matrix \
        --input-name input2 \
        --in-json-file ${feat_dt_dir}/data.json \
        --action add-scp-data-to-input \
        --verbose 1 \
        --force

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        python local/data_io.py \
            --in-scp-file data/taco_${rtask}/feats.scp \
            --ark-class matrix \
            --input-name input2 \
            --in-json-file $feat_recog_dir/data.json \
            --action add-scp-data-to-input \
            --verbose 1 \
            --force
    done

fi

if [ ${stage} -le 4 ]; then
    echo "stage 4: x-vector extraction"
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    # FIXME: This should take and scp as input
    for name in ${train_set} ${train_dev} test_clean test_other dev_clean dev_other; do
        printf "make_fbank: \033[34m${name}\033[0m\n"    
        utils/copy_data_dir.sh data/${name} data/${name}_mfcc
        steps/make_mfcc.sh \
            --write-utt2num-frames true \
            --mfcc-config conf/mfcc.conf \
            --nj 40 --cmd "$train_cmd" \
            data/${name}_mfcc exp/make_mfcc $mfccdir
        utils/fix_data_dir.sh data/${name}_mfcc
        sid/compute_vad_decision.sh --nj 41 --cmd "$train_cmd" \
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
    for name in ${train_set} ${train_dev} test_clean test_other dev_clean dev_other; do
        printf "x-vector: \033[34m${name}\033[0m\n"    
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 40 \
            $nnet_dir data/${name}_mfcc \
            $nnet_dir/xvectors_${name}
    done

    # Update json
    python local/data_io.py \
        --in-scp-file ${nnet_dir}/xvectors_${train_set}/xvector.scp \
        --ark-class vector \
        --input-name input3 \
        --in-json-file ${feat_tr_dir}/data.json \
        --action add-scp-data-to-input \
        --verbose 1 \
        --force

    python local/data_io.py \
        --in-scp-file ${nnet_dir}/xvectors_${train_dev}/xvector.scp \
        --ark-class vector \
        --input-name input3 \
        --in-json-file ${feat_dt_dir}/data.json \
        --action add-scp-data-to-input \
        --verbose 1 \
        --force

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        python local/data_io.py \
            --in-scp-file ${nnet_dir}/xvectors_${rtask}/xvector.scp \
            --ark-class vector \
            --input-name input3 \
            --in-json-file $feat_recog_dir/data.json \
            --action add-scp-data-to-input \
            --verbose 1 \
            --force
    done

fi

# You can skip this and remove --rnnlm option in the recognition (stage 5)
lmexpdir=exp/train_rnnlm_2layer_bs256
mkdir -p ${lmexpdir}
if [ ${stage} -le 5 ]; then
    echo "stage 5: LM Preparation"
    lmdatadir=data/local/lm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 data/${train_dev}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    ${cuda_cmd} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --epoch 60 \
        --batchsize 256 \
        --dict ${dict}
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

if [ ${stage} -le 6 ]; then
    echo "stage 6: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
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
        --expected-loss-conf tts \
        --asr-model $asr_model
        --asr-model-conf $asr_model_conf
fi

if [ ${stage} -le 7 ]; then
    echo "stage 7: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json 

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --lm-weight ${lm_weight} \
            &
        wait

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

