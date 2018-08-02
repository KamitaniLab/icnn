#!/bin/bash
#
# Download example data for icnn
#

## Functions

function download_file () {
    dlurl=$1
    dlpath=$2
    dldir=$(dirname $dlpath)
    dlfile=$(basename $dlpath)

    [ -d $didir ] || mkdir $dldir
    if [ -f $dldir/$dlfile ]; then
        echo "$dlfile has already been downloaded."
    else
        curl -o $dldir/$dlfile $dlurl
    fi
}

## Main

download_file https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/12565685/decoded_vgg19_cnn_feat.mat decoded_vgg19_cnn_feat.mat
download_file https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/12565688/estimated_vgg19_cnn_feat_std.mat estimated_vgg19_cnn_feat_std.mat
download_file https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/12565694/ilsvrc_2012_mean.npy ilsvrc_2012_mean.npy
