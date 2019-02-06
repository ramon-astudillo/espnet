#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=4       # start from -1 if you need to start from data download
gpu=            # will be deprecated, please use ngpu
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false # true when using CNN

# network archtecture
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
mtlalpha=0.0

# minibatch related
batchsize=12
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# tacotron loss
mdldir=train_100_pytorch_blstmp_e8_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampprob0.0_bs40_mli800_mlo150
asr_model_conf=exp/$mdldir/results/model.json
asr_model=exp/$mdldir/results/model.acc.best
tts_model_conf=exp/train_100_taco2_h8_states_enc512-3x5x512-1x512_dec2x1024_pre2x256_post5x5x512_tanh_att128-15x32_cm_bn_cc_msk_pw1.0_do0.5_zo0.1_lr1e-3_ep1e-6_wd0.0_bs50_sort_by_input_mli150_mlo400/results/model.conf
tts_model=exp/train_100_taco2_h8_states_enc512-3x5x512-1x512_dec2x1024_pre2x256_post5x5x512_tanh_att128-15x32_cm_bn_cc_msk_pw1.0_do0.5_zo0.1_lr1e-3_ep1e-6_wd0.0_bs50_sort_by_input_mli150_mlo400/results/model.loss.best

# rnnlm related
use_wordlm=false
use_rnnlm=false
lm_weight=0.2
lm_vocabsize=65000  # effective only for word LMs
lm_layers=1         # 2 for character LMs
lm_units=1000       # 650 for character LMs
lm_opt=sgd          # adam for character LMs
lm_batchsize=300    # 1024 for character LMs
lm_epochs=20        # number of epochs
lm_maxlen=40        # 150 for character LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs


# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.8
minlenratio=0.2
ctc_weight=0.0
recog_model=loss.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

n_samples=5
sample_topk=5
sample_scaling=0.1
teacher_weight=0.05
policy_gradient=true
freeze_encoder=false
unpair=audio

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
#datadir=/export/a15/vpanayotov/data
datadir=/mm0/thori/work/DB/data

# base url for downloads.
data_url=www.openslr.org/resources/12

# exp tag
tag="" # tag for managing experiments.

train_opts=""
#storage dir
scratch=/mnt/scratch06/tmp/baskar/espnetv3 

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

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#train_set=train_960
#train_set=train_100
train_set=train_460
train_paired=train_100
train_unpaired=train_360
train_dev=dev
recog_set="test_clean dev_clean"

if [ ${stage} -le -1 ]; then
    echo "stage -1: Data Download"
    for part in dev-clean test-clean train-clean-100 train-clean-360; do
        local/download_and_untar.sh ${datadir} ${data_url} ${part}
    done
fi

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean train-clean-100 train-clean-360; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/$(echo ${part} | sed s/-/_/g)
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_tr_paired_dir=${dumpdir}/${train_paired}/delta${do_delta}; mkdir -p ${feat_tr_paired_dir}
feat_tr_unpaired_dir=${dumpdir}/${train_unpaired}/delta${do_delta}; mkdir -p ${feat_tr_unpaired_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    if [[ $(hostname -f) == *.fit.vutbr.cz ]]; then
        local/make_symlink_dir.sh --tmp-root $scratch/egs/librispeech/asr1/$fbankdir $fbankdir # for BUT
    fi
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in dev_clean test_clean train_clean_100 train_clean_360; do
        #if [ ! -d ${fbankdir} ]; then
            steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 data/${x} exp/make_fbank/${x} ${fbankdir}
        #fi
    done

    #utils/combine_data.sh data/${train_set}_org data/train_clean_100 data/train_clean_360 data/train_other_500
    utils/combine_data.sh data/${train_set}_org data/train_clean_100 data/train_clean_360
    #utils/combine_data.sh data/${train_dev}_org data/dev_clean data/dev_other
    utils/combine_data.sh data/${train_dev}_org data/dev_clean

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 1500 --maxchars 300 data/${train_set}_org data/${train_set}
    remove_longshortdata.sh --maxframes 1500 --maxchars 300 data/${train_dev}_org data/${train_dev}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    elif [[ $(hostname -f) == *.fit.vutbr.cz ]]; then
        local/make_symlink_dir.sh --tmp-root $scratch/egs/librispeech/asr1/dump/${train_set}/delta${do_delta}/storage \
            ${feat_tr_dir}/storage
    fi

    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    elif [[ $(hostname -f) == *.fit.vutbr.cz ]]; then
        local/make_symlink_dir.sh --tmp-root $scratch/egs/librispeech/asr1/dump/${train_set}/delta${do_delta}/storage \
            ${feat_tr_dir}/storage
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
nlsyms=data/lang_1char/non_lang_syms.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    if [ ! -s ${dict} ]; then
        mkdir -p data/lang_1char/
        echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    fi
    wc -l ${dict}
    
    # make json labels
    if [ ! -s  ${feat_tr_dir}/data.json ]; then
      data2json.sh --feat ${feat_tr_dir}/feats.scp \
             data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    fi
    if [ ! -s  ${feat_tr_unpaired_dir}/data_text.json ]; then
        data2json.sh --feat ${feat_tr_unpaired_dir}/feats.scp --unpaired "feat" \
             data/${train_unpaired} ${dict} > ${feat_tr_unpaired_dir}/data_clean_360_audio.json
        data2json.sh --feat ${feat_tr_unpaired_dir}/feats.scp --unpaired "text" \
             data/${train_unpaired} ${dict} > ${feat_tr_unpaired_dir}/data_clean_360_text.json

    fi
    if [ ! -s  ${feat_tr_paired_dir}/data.json ]; then
        data2json.sh --feat ${feat_tr_paired_dir}/feats.scp \
             data/${train_paired} ${dict} > ${feat_tr_paired_dir}/data_clean_100.json
    fi
    if [ ! -s  ${feat_dt_dir}/data.json ]; then
        data2json.sh --feat ${feat_dt_dir}/feats.scp \
             data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    fi
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        if [ ! -s  ${feat_recog_dir}/data.json ]; then
            data2json.sh --feat ${feat_recog_dir}/feats.scp \
                data/${rtask} ${dict} > ${feat_recog_dir}/data.json
        fi
    done
fi

# It takes about one day. If you just want to do end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}
    if [ $use_wordlm = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpdir=exp/train_rnnlm_${backend}_${lmtag}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"
    
    if [ $use_wordlm = true ]; then
        lmdatadir=data/local/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        mkdir -p ${lmdatadir}
        cat data/${train_set}/text | cut -f 2- -d" " > ${lmdatadir}/train.txt
        cat data/${train_dev}/text | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        cat data/${train_test}/text | cut -f 2- -d" " > ${lmdatadir}/test.txt
        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
        lmdatadir=data/local/lm_train
        lmdict=$dict
        mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text \
            | cut -f 2- -d" " > ${lmdatadir}/train.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text \
            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/test_clean/text \
                | cut -f 2- -d" " > ${lmdatadir}/test.txt
    fi

    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --test-label ${lmdatadir}/test.txt \
        --resume ${lm_resume} \
        --layer ${lm_layers} \
        --unit ${lm_units} \
        --opt ${lm_opt} \
        --batchsize ${lm_batchsize} \
        --epoch ${lm_epochs} \
        --maxlen ${lm_maxlen} \
        --dict ${lmdict}
fi


if [ $unpair == 'audio' ]; then
    tr_json_list="${feat_tr_unpaired_dir}/data_audio.json ${feat_tr_paired_dir}/data.json"
elif [ $unpair == 'text' ]; then
    tr_json_list="${feat_tr_unpaired_dir}/data_text.json ${feat_tr_paired_dir}/data.json"
elif [ $unpair == 'audio_alone' ]; then
    tr_json_list=${feat_tr_unpaired_dir}/data_audio.json
elif [ $unpair == 'text_alone' ]; then
    tr_json_list=${feat_tr_unpaired_dir}/data_text.json
elif [ $unpair == 'audio_text_alone' ]; then
    tr_json_list="${feat_tr_unpaired_dir}/data_audio.json ${feat_tr_unpaired_dir}/data_text.json"
elif [ $unpair == 'audio_text' ]; then
    tr_json_list="${feat_tr_unpaired_dir}/data_audio.json ${feat_tr_unpaired_dir}/data_text.json ${feat_tr_paired_dir}/data.json"
fi


if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_top${sample_topk}_s${sample_scaling}_ns${n_samples}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${tag}
fi

if [ "$policy_gradient" = "true" ]; then
    expdir=${expdir}_exploss_pgrad
    train_opts="$train_opts --policy-gradient"
else
    expdir=${expdir}_exploss
fi
if [ "$freeze_encoder" = "true" ]; then
    expdir=${expdir}_freezeenc
    train_opts="$train_opts --freeze encatt"
fi

mkdir -p ${expdir}

if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_cyc_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${tr_json_list} \
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
        --asr-model-conf $asr_model_conf \
        --asr-model $asr_model \
        --tts-model-conf $tts_model_conf \
        --tts-model $tts_model \
        --expected-loss tts \
        --criterion acc \
        --update-asr-only \
        --sample-topk $sample_topk \
        --sample-scaling $sample_scaling \
        --teacher-weight $teacher_weight \
        --n-samples-per-input $n_samples \
        $train_opts
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        recog_opts=
        if [ $use_rnnlm = true ]; then
            decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}
            recog_opts="$recog_opts --lm-weight ${lm_weight} --rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}
        fi
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
            --model-conf ${expdir}/results/model.json  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            $recog_opts \
            &
        wait

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

