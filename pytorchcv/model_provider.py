from .models.alexnet import alexnet, alexnetb
from .models.zfnet import zfnet, zfnetb
from .models.vgg import (vgg11, vgg13, vgg16, vgg19, bn_vgg11, bn_vgg13, bn_vgg16, bn_vgg19, bn_vgg11b, bn_vgg13b,
                         bn_vgg16b, bn_vgg19b)
from .models.bninception import bninception
from .models.resnet import (resnet10, resnet12, resnet14, resnetbc14b, resnet16, resnet18_wd4, resnet18_wd2,
                            resnet18_w3d4, resnet18, resnet26, resnetbc26b, resnet34, resnetbc38b, resnet50, resnet50b,
                            resnet101, resnet101b, resnet152, resnet152b, resnet200, resnet200b)
from .models.preresnet import (preresnet10, preresnet12, preresnet14, preresnetbc14b, preresnet16, preresnet18_wd4,
                               preresnet18_wd2, preresnet18_w3d4, preresnet18, preresnet26, preresnetbc26b, preresnet34,
                               preresnetbc38b, preresnet50, preresnet50b, preresnet101, preresnet101b, preresnet152,
                               preresnet152b, preresnet200, preresnet200b, preresnet269b)
from .models.resnext import (resnext14_16x4d, resnext14_32x2d, resnext14_32x4d, resnext26_16x4d, resnext26_32x2d,
                             resnext26_32x4d, resnext38_32x4d, resnext50_32x4d, resnext101_32x4d, resnext101_64x4d)
from .models.seresnet import (seresnet10, seresnet12, seresnet14, seresnet16, seresnet18, seresnet26, seresnetbc26b,
                              seresnet34, seresnetbc38b, seresnet50, seresnet50b, seresnet101, seresnet101b,
                              seresnet152, seresnet152b, seresnet200, seresnet200b)
from .models.sepreresnet import (sepreresnet10, sepreresnet12, sepreresnet14, sepreresnet16, sepreresnet18,
                                 sepreresnet26, sepreresnetbc26b, sepreresnet34, sepreresnetbc38b, sepreresnet50,
                                 sepreresnet50b, sepreresnet101, sepreresnet101b, sepreresnet152, sepreresnet152b,
                                 sepreresnet200, sepreresnet200b)
from .models.seresnext import seresnext50_32x4d, seresnext101_32x4d, seresnext101_64x4d
from .models.senet import senet16, senet28, senet40, senet52, senet103, senet154
from .models.resnesta import (resnestabc14, resnesta18, resnestabc26, resnesta50, resnesta101, resnesta152, resnesta200,
                              resnesta269)
from .models.ibnresnet import ibn_resnet50, ibn_resnet101, ibn_resnet152
from .models.ibnbresnet import ibnb_resnet50, ibnb_resnet101, ibnb_resnet152
from .models.ibnresnext import ibn_resnext50_32x4d, ibn_resnext101_32x4d, ibn_resnext101_64x4d
from .models.ibndensenet import ibn_densenet121, ibn_densenet161, ibn_densenet169, ibn_densenet201
from .models.airnet import airnet50_1x64d_r2, airnet50_1x64d_r16, airnet101_1x64d_r2
from .models.airnext import airnext50_32x4d_r2, airnext101_32x4d_r2, airnext101_32x4d_r16
from .models.bamresnet import bam_resnet18, bam_resnet34, bam_resnet50, bam_resnet101, bam_resnet152
from .models.cbamresnet import cbam_resnet18, cbam_resnet34, cbam_resnet50, cbam_resnet101, cbam_resnet152
from .models.resattnet import (resattnet56, resattnet92, resattnet128, resattnet164, resattnet200, resattnet236,
                               resattnet452)
from .models.sknet import sknet50, sknet101, sknet152
from .models.scnet import scnet50, scnet101, scneta50, scneta101
from .models.regnet import (regnetx002, regnetx004, regnetx006, regnetx008, regnetx016, regnetx032, regnetx040,
                            regnetx064, regnetx080, regnetx120, regnetx160, regnetx320, regnety002, regnety004,
                            regnety006, regnety008, regnety016, regnety032, regnety040, regnety064, regnety080,
                            regnety120, regnety160, regnety320)
from .models.diaresnet import (diaresnet10, diaresnet12, diaresnet14, diaresnetbc14b, diaresnet16, diaresnet18,
                               diaresnet26, diaresnetbc26b, diaresnet34, diaresnetbc38b, diaresnet50, diaresnet50b,
                               diaresnet101, diaresnet101b, diaresnet152, diaresnet152b, diaresnet200, diaresnet200b)
from .models.diapreresnet import (diapreresnet10, diapreresnet12, diapreresnet14, diapreresnetbc14b, diapreresnet16,
                                  diapreresnet18, diapreresnet26, diapreresnetbc26b, diapreresnet34, diapreresnetbc38b,
                                  diapreresnet50, diapreresnet50b, diapreresnet101, diapreresnet101b, diapreresnet152,
                                  diapreresnet152b, diapreresnet200, diapreresnet200b, diapreresnet269b)
from .models.pyramidnet import pyramidnet101_a360
from .models.diracnetv2 import diracnet18v2, diracnet34v2
from .models.sharesnet import (sharesnet18, sharesnet34, sharesnet50, sharesnet50b, sharesnet101, sharesnet101b,
                               sharesnet152, sharesnet152b)
from .models.densenet import densenet121, densenet161, densenet169, densenet201
from .models.condensenet import condensenet74_c4_g4, condensenet74_c8_g8
from .models.sparsenet import sparsenet121, sparsenet161, sparsenet169, sparsenet201, sparsenet264
from .models.peleenet import peleenet
from .models.wrn import wrn50_2
from .models.drn import drnc26, drnc42, drnc58, drnd22, drnd38, drnd54, drnd105
from .models.dpn import dpn68, dpn68b, dpn98, dpn107, dpn131
from .models.darknet import darknet_ref, darknet_tiny, darknet19
from .models.darknet53 import darknet53
from .models.channelnet import channelnet
from .models.isqrtcovresnet import (isqrtcovresnet18, isqrtcovresnet34, isqrtcovresnet50, isqrtcovresnet50b,
                                    isqrtcovresnet101, isqrtcovresnet101b)
from .models.revnet import revnet38, revnet110, revnet164
from .models.irevnet import irevnet301
from .models.bagnet import bagnet9, bagnet17, bagnet33
from .models.dla import dla34, dla46c, dla46xc, dla60, dla60x, dla60xc, dla102, dla102x, dla102x2, dla169
from .models.msdnet import msdnet22
from .models.fishnet import fishnet99, fishnet150
from .models.espnetv2 import espnetv2_wd2, espnetv2_w1, espnetv2_w5d4, espnetv2_w3d2, espnetv2_w2
from .models.dicenet import (dicenet_wd5, dicenet_wd2, dicenet_w3d4, dicenet_w1, dicenet_w5d4, dicenet_w3d2,
                             dicenet_w7d8, dicenet_w2)
from .models.hrnet import (hrnet_w18_small_v1, hrnet_w18_small_v2, hrnetv2_w18, hrnetv2_w30, hrnetv2_w32, hrnetv2_w40,
                           hrnetv2_w44, hrnetv2_w48, hrnetv2_w64)
from .models.vovnet import vovnet27s, vovnet39, vovnet57
from .models.selecsls import selecsls42, selecsls42b, selecsls60, selecsls60b, selecsls84
from .models.hardnet import hardnet39ds, hardnet68ds, hardnet68, hardnet85
from .models.xdensenet import xdensenet121_2, xdensenet161_2, xdensenet169_2, xdensenet201_2
from .models.squeezenet import squeezenet_v1_0, squeezenet_v1_1, squeezeresnet_v1_0, squeezeresnet_v1_1
from .models.squeezenext import sqnxt23_w1, sqnxt23_w3d2, sqnxt23_w2, sqnxt23v5_w1, sqnxt23v5_w3d2, sqnxt23v5_w2
from .models.shufflenet import (shufflenet_g1_w1, shufflenet_g2_w1, shufflenet_g3_w1, shufflenet_g4_w1,
                                shufflenet_g8_w1, shufflenet_g1_w3d4, shufflenet_g3_w3d4, shufflenet_g1_wd2,
                                shufflenet_g3_wd2, shufflenet_g1_wd4, shufflenet_g3_wd4)
from .models.shufflenetv2 import shufflenetv2_wd2, shufflenetv2_w1, shufflenetv2_w3d2, shufflenetv2_w2
from .models.shufflenetv2b import shufflenetv2b_wd2, shufflenetv2b_w1, shufflenetv2b_w3d2, shufflenetv2b_w2
from .models.menet import (menet108_8x1_g3, menet128_8x1_g4, menet160_8x1_g8, menet228_12x1_g3, menet256_12x1_g4,
                           menet348_12x1_g3, menet352_12x1_g8, menet456_24x1_g3)
from .models.mobilenet import mobilenet_w1, mobilenet_w3d4, mobilenet_wd2, mobilenet_wd4
from .models.mobilenetb import mobilenetb_w1, mobilenetb_w3d4, mobilenetb_wd2, mobilenetb_wd4
from .models.fdmobilenet import fdmobilenet_w1, fdmobilenet_w3d4, fdmobilenet_wd2, fdmobilenet_wd4
from .models.mobilenetv2 import (mobilenetv2_w1, mobilenetv2_w3d4, mobilenetv2_wd2, mobilenetv2_wd4, mobilenetv2b_w1,
                                 mobilenetv2b_w3d4, mobilenetv2b_wd2, mobilenetv2b_wd4)
from .models.mobilenetv3 import (mobilenetv3_small_w7d20, mobilenetv3_small_wd2, mobilenetv3_small_w3d4,
                                 mobilenetv3_small_w1, mobilenetv3_small_w5d4, mobilenetv3_large_w7d20,
                                 mobilenetv3_large_wd2, mobilenetv3_large_w3d4, mobilenetv3_large_w1,
                                 mobilenetv3_large_w5d4)
from .models.igcv3 import igcv3_w1, igcv3_w3d4, igcv3_wd2, igcv3_wd4
from .models.ghostnet import ghostnet
from .models.mnasnet import mnasnet_b1, mnasnet_a1, mnasnet_small
from .models.darts import darts
from .models.proxylessnas import proxylessnas_cpu, proxylessnas_gpu, proxylessnas_mobile, proxylessnas_mobile14
from .models.fbnet import fbnet_cb
from .models.xception import xception
from .models.inceptionv3 import inceptionv3
from .models.inceptionv4 import inceptionv4
from .models.inceptionresnetv1 import inceptionresnetv1
from .models.inceptionresnetv2 import inceptionresnetv2
from .models.polynet import polynet
from .models.nasnet import nasnet_4a1056, nasnet_6a4032
from .models.pnasnet import pnasnet5large
from .models.spnasnet import spnasnet
from .models.efficientnet import (efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4,
                                  efficientnet_b5, efficientnet_b6, efficientnet_b7, efficientnet_b8, efficientnet_b0b,
                                  efficientnet_b1b, efficientnet_b2b, efficientnet_b3b, efficientnet_b4b,
                                  efficientnet_b5b, efficientnet_b6b, efficientnet_b7b, efficientnet_b0c,
                                  efficientnet_b1c, efficientnet_b2c, efficientnet_b3c, efficientnet_b4c,
                                  efficientnet_b5c, efficientnet_b6c, efficientnet_b7c, efficientnet_b8c)
from .models.efficientnetedge import efficientnet_edge_small_b, efficientnet_edge_medium_b, efficientnet_edge_large_b
from .models.mixnet import mixnet_s, mixnet_m, mixnet_l

from .models.nin_cifar import nin_cifar10, nin_cifar100, nin_svhn
from .models.resnet_cifar import (resnet20_cifar10, resnet20_cifar100, resnet20_svhn, resnet56_cifar10,
                                  resnet56_cifar100, resnet56_svhn, resnet110_cifar10, resnet110_cifar100,
                                  resnet110_svhn, resnet164bn_cifar10, resnet164bn_cifar100, resnet164bn_svhn,
                                  resnet272bn_cifar10, resnet272bn_cifar100, resnet272bn_svhn, resnet542bn_cifar10,
                                  resnet542bn_cifar100, resnet542bn_svhn, resnet1001_cifar10, resnet1001_cifar100,
                                  resnet1001_svhn, resnet1202_cifar10, resnet1202_cifar100, resnet1202_svhn)
from .models.preresnet_cifar import (preresnet20_cifar10, preresnet20_cifar100, preresnet20_svhn, preresnet56_cifar10,
                                     preresnet56_cifar100, preresnet56_svhn, preresnet110_cifar10,
                                     preresnet110_cifar100, preresnet110_svhn, preresnet164bn_cifar10,
                                     preresnet164bn_cifar100, preresnet164bn_svhn, preresnet272bn_cifar10,
                                     preresnet272bn_cifar100, preresnet272bn_svhn, preresnet542bn_cifar10,
                                     preresnet542bn_cifar100, preresnet542bn_svhn, preresnet1001_cifar10,
                                     preresnet1001_cifar100, preresnet1001_svhn, preresnet1202_cifar10,
                                     preresnet1202_cifar100, preresnet1202_svhn)
from .models.resnext_cifar import (resnext20_16x4d_cifar10, resnext20_16x4d_cifar100, resnext20_16x4d_svhn,
                                   resnext20_32x2d_cifar10, resnext20_32x2d_cifar100, resnext20_32x2d_svhn,
                                   resnext20_32x4d_cifar10, resnext20_32x4d_cifar100, resnext20_32x4d_svhn,
                                   resnext29_32x4d_cifar10, resnext29_32x4d_cifar100, resnext29_32x4d_svhn,
                                   resnext29_16x64d_cifar10, resnext29_16x64d_cifar100, resnext29_16x64d_svhn,
                                   resnext272_1x64d_cifar10, resnext272_1x64d_cifar100, resnext272_1x64d_svhn,
                                   resnext272_2x32d_cifar10, resnext272_2x32d_cifar100, resnext272_2x32d_svhn)
from .models.seresnet_cifar import (seresnet20_cifar10, seresnet20_cifar100, seresnet20_svhn, seresnet56_cifar10,
                                    seresnet56_cifar100, seresnet56_svhn, seresnet110_cifar10, seresnet110_cifar100,
                                    seresnet110_svhn, seresnet164bn_cifar10, seresnet164bn_cifar100, seresnet164bn_svhn,
                                    seresnet272bn_cifar10, seresnet272bn_cifar100, seresnet272bn_svhn,
                                    seresnet542bn_cifar10, seresnet542bn_cifar100, seresnet542bn_svhn,
                                    seresnet1001_cifar10, seresnet1001_cifar100, seresnet1001_svhn,
                                    seresnet1202_cifar10, seresnet1202_cifar100, seresnet1202_svhn)
from .models.sepreresnet_cifar import (sepreresnet20_cifar10, sepreresnet20_cifar100, sepreresnet20_svhn,
                                       sepreresnet56_cifar10, sepreresnet56_cifar100, sepreresnet56_svhn,
                                       sepreresnet110_cifar10, sepreresnet110_cifar100, sepreresnet110_svhn,
                                       sepreresnet164bn_cifar10, sepreresnet164bn_cifar100, sepreresnet164bn_svhn,
                                       sepreresnet272bn_cifar10, sepreresnet272bn_cifar100, sepreresnet272bn_svhn,
                                       sepreresnet542bn_cifar10, sepreresnet542bn_cifar100, sepreresnet542bn_svhn,
                                       sepreresnet1001_cifar10, sepreresnet1001_cifar100, sepreresnet1001_svhn,
                                       sepreresnet1202_cifar10, sepreresnet1202_cifar100, sepreresnet1202_svhn)
from .models.pyramidnet_cifar import (pyramidnet110_a48_cifar10, pyramidnet110_a48_cifar100, pyramidnet110_a48_svhn,
                                      pyramidnet110_a84_cifar10, pyramidnet110_a84_cifar100, pyramidnet110_a84_svhn,
                                      pyramidnet110_a270_cifar10, pyramidnet110_a270_cifar100, pyramidnet110_a270_svhn,
                                      pyramidnet164_a270_bn_cifar10, pyramidnet164_a270_bn_cifar100,
                                      pyramidnet164_a270_bn_svhn, pyramidnet200_a240_bn_cifar10,
                                      pyramidnet200_a240_bn_cifar100, pyramidnet200_a240_bn_svhn,
                                      pyramidnet236_a220_bn_cifar10, pyramidnet236_a220_bn_cifar100,
                                      pyramidnet236_a220_bn_svhn, pyramidnet272_a200_bn_cifar10,
                                      pyramidnet272_a200_bn_cifar100, pyramidnet272_a200_bn_svhn)
from .models.densenet_cifar import (densenet40_k12_cifar10, densenet40_k12_cifar100, densenet40_k12_svhn,
                                    densenet40_k12_bc_cifar10, densenet40_k12_bc_cifar100, densenet40_k12_bc_svhn,
                                    densenet40_k24_bc_cifar10, densenet40_k24_bc_cifar100, densenet40_k24_bc_svhn,
                                    densenet40_k36_bc_cifar10, densenet40_k36_bc_cifar100, densenet40_k36_bc_svhn,
                                    densenet100_k12_cifar10, densenet100_k12_cifar100, densenet100_k12_svhn,
                                    densenet100_k24_cifar10, densenet100_k24_cifar100, densenet100_k24_svhn,
                                    densenet100_k12_bc_cifar10, densenet100_k12_bc_cifar100, densenet100_k12_bc_svhn,
                                    densenet190_k40_bc_cifar10, densenet190_k40_bc_cifar100, densenet190_k40_bc_svhn,
                                    densenet250_k24_bc_cifar10, densenet250_k24_bc_cifar100, densenet250_k24_bc_svhn)
from .models.xdensenet_cifar import (xdensenet40_2_k24_bc_cifar10, xdensenet40_2_k24_bc_cifar100,
                                     xdensenet40_2_k24_bc_svhn, xdensenet40_2_k36_bc_cifar10,
                                     xdensenet40_2_k36_bc_cifar100, xdensenet40_2_k36_bc_svhn)
from .models.wrn_cifar import (wrn16_10_cifar10, wrn16_10_cifar100, wrn16_10_svhn, wrn28_10_cifar10, wrn28_10_cifar100,
                               wrn28_10_svhn, wrn40_8_cifar10, wrn40_8_cifar100, wrn40_8_svhn)
from .models.wrn1bit_cifar import (wrn20_10_1bit_cifar10, wrn20_10_1bit_cifar100, wrn20_10_1bit_svhn,
                                   wrn20_10_32bit_cifar10, wrn20_10_32bit_cifar100, wrn20_10_32bit_svhn)
from .models.ror_cifar import (ror3_56_cifar10, ror3_56_cifar100, ror3_56_svhn, ror3_110_cifar10, ror3_110_cifar100,
                               ror3_110_svhn, ror3_164_cifar10, ror3_164_cifar100, ror3_164_svhn)
from .models.rir_cifar import rir_cifar10, rir_cifar100, rir_svhn
from .models.msdnet_cifar10 import msdnet22_cifar10
from .models.resdropresnet_cifar import resdropresnet20_cifar10, resdropresnet20_cifar100, resdropresnet20_svhn
from .models.shakeshakeresnet_cifar import (shakeshakeresnet20_2x16d_cifar10, shakeshakeresnet20_2x16d_cifar100,
                                            shakeshakeresnet20_2x16d_svhn, shakeshakeresnet26_2x32d_cifar10,
                                            shakeshakeresnet26_2x32d_cifar100, shakeshakeresnet26_2x32d_svhn)
from .models.shakedropresnet_cifar import shakedropresnet20_cifar10, shakedropresnet20_cifar100, shakedropresnet20_svhn
from .models.fractalnet_cifar import fractalnet_cifar10, fractalnet_cifar100
from .models.diaresnet_cifar import (diaresnet20_cifar10, diaresnet20_cifar100, diaresnet20_svhn, diaresnet56_cifar10,
                                     diaresnet56_cifar100, diaresnet56_svhn, diaresnet110_cifar10,
                                     diaresnet110_cifar100, diaresnet110_svhn, diaresnet164bn_cifar10,
                                     diaresnet164bn_cifar100, diaresnet164bn_svhn, diaresnet1001_cifar10,
                                     diaresnet1001_cifar100, diaresnet1001_svhn, diaresnet1202_cifar10,
                                     diaresnet1202_cifar100, diaresnet1202_svhn)
from .models.diapreresnet_cifar import (diapreresnet20_cifar10, diapreresnet20_cifar100, diapreresnet20_svhn,
                                        diapreresnet56_cifar10, diapreresnet56_cifar100, diapreresnet56_svhn,
                                        diapreresnet110_cifar10, diapreresnet110_cifar100, diapreresnet110_svhn,
                                        diapreresnet164bn_cifar10, diapreresnet164bn_cifar100, diapreresnet164bn_svhn,
                                        diapreresnet1001_cifar10, diapreresnet1001_cifar100, diapreresnet1001_svhn,
                                        diapreresnet1202_cifar10, diapreresnet1202_cifar100, diapreresnet1202_svhn)

from .models.octresnet import octresnet10_ad2, octresnet50b_ad2

from .models.resneta import resneta10, resnetabc14b, resneta18, resneta50b, resneta101b, resneta152b
from .models.resnetd import resnetd50b, resnetd101b, resnetd152b
from .models.fastseresnet import fastseresnet101b

from .models.resnet_cub import (resnet10_cub, resnet12_cub, resnet14_cub, resnetbc14b_cub, resnet16_cub, resnet18_cub,
                                resnet26_cub, resnetbc26b_cub, resnet34_cub, resnetbc38b_cub, resnet50_cub,
                                resnet50b_cub, resnet101_cub, resnet101b_cub, resnet152_cub, resnet152b_cub,
                                resnet200_cub, resnet200b_cub)
from .models.seresnet_cub import (seresnet10_cub, seresnet12_cub, seresnet14_cub, seresnetbc14b_cub, seresnet16_cub,
                                  seresnet18_cub, seresnet26_cub, seresnetbc26b_cub, seresnet34_cub, seresnetbc38b_cub,
                                  seresnet50_cub, seresnet50b_cub, seresnet101_cub, seresnet101b_cub, seresnet152_cub,
                                  seresnet152b_cub, seresnet200_cub, seresnet200b_cub)
from .models.mobilenet_cub import (mobilenet_w1_cub, mobilenet_w3d4_cub, mobilenet_wd2_cub, mobilenet_wd4_cub,
                                   fdmobilenet_w1_cub, fdmobilenet_w3d4_cub, fdmobilenet_wd2_cub, fdmobilenet_wd4_cub)
from .models.proxylessnas_cub import (proxylessnas_cpu_cub, proxylessnas_gpu_cub, proxylessnas_mobile_cub,
                                      proxylessnas_mobile14_cub)
from .models.ntsnet_cub import ntsnet_cub

from .models.fcn8sd import (fcn8sd_resnetd50b_voc, fcn8sd_resnetd101b_voc, fcn8sd_resnetd50b_coco,
                            fcn8sd_resnetd101b_coco, fcn8sd_resnetd50b_ade20k, fcn8sd_resnetd101b_ade20k,
                            fcn8sd_resnetd50b_cityscapes, fcn8sd_resnetd101b_cityscapes)
from .models.pspnet import (pspnet_resnetd50b_voc, pspnet_resnetd101b_voc, pspnet_resnetd50b_coco,
                            pspnet_resnetd101b_coco, pspnet_resnetd50b_ade20k, pspnet_resnetd101b_ade20k,
                            pspnet_resnetd50b_cityscapes, pspnet_resnetd101b_cityscapes)
from .models.deeplabv3 import (deeplabv3_resnetd50b_voc, deeplabv3_resnetd101b_voc, deeplabv3_resnetd152b_voc,
                               deeplabv3_resnetd50b_coco, deeplabv3_resnetd101b_coco, deeplabv3_resnetd152b_coco,
                               deeplabv3_resnetd50b_ade20k, deeplabv3_resnetd101b_ade20k,
                               deeplabv3_resnetd50b_cityscapes, deeplabv3_resnetd101b_cityscapes)
from .models.icnet import icnet_resnetd50b_cityscapes
from .models.fastscnn import fastscnn_cityscapes
from .models.cgnet import cgnet_cityscapes
from .models.dabnet import dabnet_cityscapes
from .models.sinet import sinet_cityscapes
from .models.bisenet import bisenet_resnet18_celebamaskhq
from .models.danet import danet_resnetd50b_cityscapes, danet_resnetd101b_cityscapes
from .models.fpenet import fpenet_cityscapes
from .models.contextnet import ctxnet_cityscapes
from .models.lednet import lednet_cityscapes
from .models.esnet import esnet_cityscapes
from .models.edanet import edanet_cityscapes
from .models.enet import enet_cityscapes
from .models.erfnet import erfnet_cityscapes
from .models.linknet import linknet_cityscapes
from .models.segnet import segnet_cityscapes
from .models.unet import unet_cityscapes
from .models.sqnet import sqnet_cityscapes

from .models.alphapose_coco import alphapose_fastseresnet101b_coco
from .models.simplepose_coco import (simplepose_resnet18_coco, simplepose_resnet50b_coco, simplepose_resnet101b_coco,
                                     simplepose_resnet152b_coco, simplepose_resneta50b_coco,
                                     simplepose_resneta101b_coco, simplepose_resneta152b_coco)
from .models.simpleposemobile_coco import (simplepose_mobile_resnet18_coco, simplepose_mobile_resnet50b_coco,
                                           simplepose_mobile_mobilenet_w1_coco, simplepose_mobile_mobilenetv2b_w1_coco,
                                           simplepose_mobile_mobilenetv3_small_w1_coco,
                                           simplepose_mobile_mobilenetv3_large_w1_coco)
from .models.lwopenpose_cmupan import lwopenpose2d_mobilenet_cmupan_coco, lwopenpose3d_mobilenet_cmupan_coco
from .models.ibppose_coco import ibppose_coco

from .models.prnet import prnet

from .models.centernet import (centernet_resnet18_voc, centernet_resnet18_coco, centernet_resnet50b_voc,
                               centernet_resnet50b_coco, centernet_resnet101b_voc, centernet_resnet101b_coco)
from .models.lffd import lffd20x5s320v2_widerface, lffd25x8s560v1_widerface

from .models.pfpcnet import pfpcnet
from .models.voca import voca8flame
from .models.nvpattexp import nvpattexp116bazel76
from .models.visemenet import visemenet20

from .models.superpointnet import superpointnet

from .models.jasper import jasper5x3, jasper10x4, jasper10x5
from .models.jasperdr import jasperdr10x5_en, jasperdr10x5_en_nr
from .models.quartznet import (quartznet5x5_en_ls, quartznet15x5_en, quartznet15x5_en_nr, quartznet15x5_fr,
                               quartznet15x5_de, quartznet15x5_it, quartznet15x5_es, quartznet15x5_ca, quartznet15x5_pl,
                               quartznet15x5_ru, quartznet15x5_ru34)

from .models.tresnet import tresnet_m, tresnet_l, tresnet_xl

from .models.espcnet import espcnet_cityscapes

from .models.raft import raft_things, raft_small
from .models.propainter_rfc import propainter_rfc
from .models.propainter_ip import propainter_ip
from .models.propainter import propainter

# from .models.others.oth_quartznet import *

# from .models.others.oth_pose_resnet import *
# from .models.others.oth_lwopenpose2d import *
# from .models.others.oth_lwopenpose3d import *
# from .models.others.oth_prnet import *
# from .models.others.oth_sinet import *
# from .models.others.oth_ibppose import *
# from .models.others.oth_bisenet1 import *
# from .models.others.oth_regnet import *

# from .models.others.oth_tresnet import *
# from .models.tresnet import *
# from .models.others.oth_dabnet import *

__all__ = ['get_model']


_models = {
    'alexnet': alexnet,
    'alexnetb': alexnetb,

    'zfnet': zfnet,
    'zfnetb': zfnetb,

    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'bn_vgg11': bn_vgg11,
    'bn_vgg13': bn_vgg13,
    'bn_vgg16': bn_vgg16,
    'bn_vgg19': bn_vgg19,
    'bn_vgg11b': bn_vgg11b,
    'bn_vgg13b': bn_vgg13b,
    'bn_vgg16b': bn_vgg16b,
    'bn_vgg19b': bn_vgg19b,

    'bninception': bninception,

    'resnet10': resnet10,
    'resnet12': resnet12,
    'resnet14': resnet14,
    'resnetbc14b': resnetbc14b,
    'resnet16': resnet16,
    'resnet18_wd4': resnet18_wd4,
    'resnet18_wd2': resnet18_wd2,
    'resnet18_w3d4': resnet18_w3d4,
    'resnet18': resnet18,
    'resnet26': resnet26,
    'resnetbc26b': resnetbc26b,
    'resnet34': resnet34,
    'resnetbc38b': resnetbc38b,
    'resnet50': resnet50,
    'resnet50b': resnet50b,
    'resnet101': resnet101,
    'resnet101b': resnet101b,
    'resnet152': resnet152,
    'resnet152b': resnet152b,
    'resnet200': resnet200,
    'resnet200b': resnet200b,

    'preresnet10': preresnet10,
    'preresnet12': preresnet12,
    'preresnet14': preresnet14,
    'preresnetbc14b': preresnetbc14b,
    'preresnet16': preresnet16,
    'preresnet18_wd4': preresnet18_wd4,
    'preresnet18_wd2': preresnet18_wd2,
    'preresnet18_w3d4': preresnet18_w3d4,
    'preresnet18': preresnet18,
    'preresnet26': preresnet26,
    'preresnetbc26b': preresnetbc26b,
    'preresnet34': preresnet34,
    'preresnetbc38b': preresnetbc38b,
    'preresnet50': preresnet50,
    'preresnet50b': preresnet50b,
    'preresnet101': preresnet101,
    'preresnet101b': preresnet101b,
    'preresnet152': preresnet152,
    'preresnet152b': preresnet152b,
    'preresnet200': preresnet200,
    'preresnet200b': preresnet200b,
    'preresnet269b': preresnet269b,

    'resnext14_16x4d': resnext14_16x4d,
    'resnext14_32x2d': resnext14_32x2d,
    'resnext14_32x4d': resnext14_32x4d,
    'resnext26_16x4d': resnext26_16x4d,
    'resnext26_32x2d': resnext26_32x2d,
    'resnext26_32x4d': resnext26_32x4d,
    'resnext38_32x4d': resnext38_32x4d,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x4d': resnext101_32x4d,
    'resnext101_64x4d': resnext101_64x4d,

    'seresnet10': seresnet10,
    'seresnet12': seresnet12,
    'seresnet14': seresnet14,
    'seresnet16': seresnet16,
    'seresnet18': seresnet18,
    'seresnet26': seresnet26,
    'seresnetbc26b': seresnetbc26b,
    'seresnet34': seresnet34,
    'seresnetbc38b': seresnetbc38b,
    'seresnet50': seresnet50,
    'seresnet50b': seresnet50b,
    'seresnet101': seresnet101,
    'seresnet101b': seresnet101b,
    'seresnet152': seresnet152,
    'seresnet152b': seresnet152b,
    'seresnet200': seresnet200,
    'seresnet200b': seresnet200b,

    'sepreresnet10': sepreresnet10,
    'sepreresnet12': sepreresnet12,
    'sepreresnet14': sepreresnet14,
    'sepreresnet16': sepreresnet16,
    'sepreresnet18': sepreresnet18,
    'sepreresnet26': sepreresnet26,
    'sepreresnetbc26b': sepreresnetbc26b,
    'sepreresnet34': sepreresnet34,
    'sepreresnetbc38b': sepreresnetbc38b,
    'sepreresnet50': sepreresnet50,
    'sepreresnet50b': sepreresnet50b,
    'sepreresnet101': sepreresnet101,
    'sepreresnet101b': sepreresnet101b,
    'sepreresnet152': sepreresnet152,
    'sepreresnet152b': sepreresnet152b,
    'sepreresnet200': sepreresnet200,
    'sepreresnet200b': sepreresnet200b,

    'seresnext50_32x4d': seresnext50_32x4d,
    'seresnext101_32x4d': seresnext101_32x4d,
    'seresnext101_64x4d': seresnext101_64x4d,

    'senet16': senet16,
    'senet28': senet28,
    'senet40': senet40,
    'senet52': senet52,
    'senet103': senet103,
    'senet154': senet154,

    'resnestabc14': resnestabc14,
    'resnesta18': resnesta18,
    'resnestabc26': resnestabc26,
    'resnesta50': resnesta50,
    'resnesta101': resnesta101,
    'resnesta152': resnesta152,
    'resnesta200': resnesta200,
    'resnesta269': resnesta269,

    'ibn_resnet50': ibn_resnet50,
    'ibn_resnet101': ibn_resnet101,
    'ibn_resnet152': ibn_resnet152,

    'ibnb_resnet50': ibnb_resnet50,
    'ibnb_resnet101': ibnb_resnet101,
    'ibnb_resnet152': ibnb_resnet152,

    'ibn_resnext50_32x4d': ibn_resnext50_32x4d,
    'ibn_resnext101_32x4d': ibn_resnext101_32x4d,
    'ibn_resnext101_64x4d': ibn_resnext101_64x4d,

    'ibn_densenet121': ibn_densenet121,
    'ibn_densenet161': ibn_densenet161,
    'ibn_densenet169': ibn_densenet169,
    'ibn_densenet201': ibn_densenet201,

    'airnet50_1x64d_r2': airnet50_1x64d_r2,
    'airnet50_1x64d_r16': airnet50_1x64d_r16,
    'airnet101_1x64d_r2': airnet101_1x64d_r2,

    'airnext50_32x4d_r2': airnext50_32x4d_r2,
    'airnext101_32x4d_r2': airnext101_32x4d_r2,
    'airnext101_32x4d_r16': airnext101_32x4d_r16,

    'bam_resnet18': bam_resnet18,
    'bam_resnet34': bam_resnet34,
    'bam_resnet50': bam_resnet50,
    'bam_resnet101': bam_resnet101,
    'bam_resnet152': bam_resnet152,

    'cbam_resnet18': cbam_resnet18,
    'cbam_resnet34': cbam_resnet34,
    'cbam_resnet50': cbam_resnet50,
    'cbam_resnet101': cbam_resnet101,
    'cbam_resnet152': cbam_resnet152,

    'resattnet56': resattnet56,
    'resattnet92': resattnet92,
    'resattnet128': resattnet128,
    'resattnet164': resattnet164,
    'resattnet200': resattnet200,
    'resattnet236': resattnet236,
    'resattnet452': resattnet452,

    'sknet50': sknet50,
    'sknet101': sknet101,
    'sknet152': sknet152,

    'scnet50': scnet50,
    'scnet101': scnet101,
    'scneta50': scneta50,
    'scneta101': scneta101,

    'regnetx002': regnetx002,
    'regnetx004': regnetx004,
    'regnetx006': regnetx006,
    'regnetx008': regnetx008,
    'regnetx016': regnetx016,
    'regnetx032': regnetx032,
    'regnetx040': regnetx040,
    'regnetx064': regnetx064,
    'regnetx080': regnetx080,
    'regnetx120': regnetx120,
    'regnetx160': regnetx160,
    'regnetx320': regnetx320,

    'regnety002': regnety002,
    'regnety004': regnety004,
    'regnety006': regnety006,
    'regnety008': regnety008,
    'regnety016': regnety016,
    'regnety032': regnety032,
    'regnety040': regnety040,
    'regnety064': regnety064,
    'regnety080': regnety080,
    'regnety120': regnety120,
    'regnety160': regnety160,
    'regnety320': regnety320,

    'diaresnet10': diaresnet10,
    'diaresnet12': diaresnet12,
    'diaresnet14': diaresnet14,
    'diaresnetbc14b': diaresnetbc14b,
    'diaresnet16': diaresnet16,
    'diaresnet18': diaresnet18,
    'diaresnet26': diaresnet26,
    'diaresnetbc26b': diaresnetbc26b,
    'diaresnet34': diaresnet34,
    'diaresnetbc38b': diaresnetbc38b,
    'diaresnet50': diaresnet50,
    'diaresnet50b': diaresnet50b,
    'diaresnet101': diaresnet101,
    'diaresnet101b': diaresnet101b,
    'diaresnet152': diaresnet152,
    'diaresnet152b': diaresnet152b,
    'diaresnet200': diaresnet200,
    'diaresnet200b': diaresnet200b,

    'diapreresnet10': diapreresnet10,
    'diapreresnet12': diapreresnet12,
    'diapreresnet14': diapreresnet14,
    'diapreresnetbc14b': diapreresnetbc14b,
    'diapreresnet16': diapreresnet16,
    'diapreresnet18': diapreresnet18,
    'diapreresnet26': diapreresnet26,
    'diapreresnetbc26b': diapreresnetbc26b,
    'diapreresnet34': diapreresnet34,
    'diapreresnetbc38b': diapreresnetbc38b,
    'diapreresnet50': diapreresnet50,
    'diapreresnet50b': diapreresnet50b,
    'diapreresnet101': diapreresnet101,
    'diapreresnet101b': diapreresnet101b,
    'diapreresnet152': diapreresnet152,
    'diapreresnet152b': diapreresnet152b,
    'diapreresnet200': diapreresnet200,
    'diapreresnet200b': diapreresnet200b,
    'diapreresnet269b': diapreresnet269b,

    'pyramidnet101_a360': pyramidnet101_a360,

    'diracnet18v2': diracnet18v2,
    'diracnet34v2': diracnet34v2,

    'sharesnet18': sharesnet18,
    'sharesnet34': sharesnet34,
    'sharesnet50': sharesnet50,
    'sharesnet50b': sharesnet50b,
    'sharesnet101': sharesnet101,
    'sharesnet101b': sharesnet101b,
    'sharesnet152': sharesnet152,
    'sharesnet152b': sharesnet152b,

    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet169': densenet169,
    'densenet201': densenet201,

    'condensenet74_c4_g4': condensenet74_c4_g4,
    'condensenet74_c8_g8': condensenet74_c8_g8,

    'sparsenet121': sparsenet121,
    'sparsenet161': sparsenet161,
    'sparsenet169': sparsenet169,
    'sparsenet201': sparsenet201,
    'sparsenet264': sparsenet264,

    'peleenet': peleenet,

    'wrn50_2': wrn50_2,

    'drnc26': drnc26,
    'drnc42': drnc42,
    'drnc58': drnc58,
    'drnd22': drnd22,
    'drnd38': drnd38,
    'drnd54': drnd54,
    'drnd105': drnd105,

    'dpn68': dpn68,
    'dpn68b': dpn68b,
    'dpn98': dpn98,
    'dpn107': dpn107,
    'dpn131': dpn131,

    'darknet_ref': darknet_ref,
    'darknet_tiny': darknet_tiny,
    'darknet19': darknet19,
    'darknet53': darknet53,

    'channelnet': channelnet,

    'revnet38': revnet38,
    'revnet110': revnet110,
    'revnet164': revnet164,

    'irevnet301': irevnet301,

    'bagnet9': bagnet9,
    'bagnet17': bagnet17,
    'bagnet33': bagnet33,

    'dla34': dla34,
    'dla46c': dla46c,
    'dla46xc': dla46xc,
    'dla60': dla60,
    'dla60x': dla60x,
    'dla60xc': dla60xc,
    'dla102': dla102,
    'dla102x': dla102x,
    'dla102x2': dla102x2,
    'dla169': dla169,

    'msdnet22': msdnet22,

    'fishnet99': fishnet99,
    'fishnet150': fishnet150,

    'espnetv2_wd2': espnetv2_wd2,
    'espnetv2_w1': espnetv2_w1,
    'espnetv2_w5d4': espnetv2_w5d4,
    'espnetv2_w3d2': espnetv2_w3d2,
    'espnetv2_w2': espnetv2_w2,

    'dicenet_wd5': dicenet_wd5,
    'dicenet_wd2': dicenet_wd2,
    'dicenet_w3d4': dicenet_w3d4,
    'dicenet_w1': dicenet_w1,
    'dicenet_w5d4': dicenet_w5d4,
    'dicenet_w3d2': dicenet_w3d2,
    'dicenet_w7d8': dicenet_w7d8,
    'dicenet_w2': dicenet_w2,

    'hrnet_w18_small_v1': hrnet_w18_small_v1,
    'hrnet_w18_small_v2': hrnet_w18_small_v2,
    'hrnetv2_w18': hrnetv2_w18,
    'hrnetv2_w30': hrnetv2_w30,
    'hrnetv2_w32': hrnetv2_w32,
    'hrnetv2_w40': hrnetv2_w40,
    'hrnetv2_w44': hrnetv2_w44,
    'hrnetv2_w48': hrnetv2_w48,
    'hrnetv2_w64': hrnetv2_w64,

    'vovnet27s': vovnet27s,
    'vovnet39': vovnet39,
    'vovnet57': vovnet57,

    'selecsls42': selecsls42,
    'selecsls42b': selecsls42b,
    'selecsls60': selecsls60,
    'selecsls60b': selecsls60b,
    'selecsls84': selecsls84,

    'hardnet39ds': hardnet39ds,
    'hardnet68ds': hardnet68ds,
    'hardnet68': hardnet68,
    'hardnet85': hardnet85,

    'xdensenet121_2': xdensenet121_2,
    'xdensenet161_2': xdensenet161_2,
    'xdensenet169_2': xdensenet169_2,
    'xdensenet201_2': xdensenet201_2,

    'squeezenet_v1_0': squeezenet_v1_0,
    'squeezenet_v1_1': squeezenet_v1_1,

    'squeezeresnet_v1_0': squeezeresnet_v1_0,
    'squeezeresnet_v1_1': squeezeresnet_v1_1,

    'sqnxt23_w1': sqnxt23_w1,
    'sqnxt23_w3d2': sqnxt23_w3d2,
    'sqnxt23_w2': sqnxt23_w2,
    'sqnxt23v5_w1': sqnxt23v5_w1,
    'sqnxt23v5_w3d2': sqnxt23v5_w3d2,
    'sqnxt23v5_w2': sqnxt23v5_w2,

    'shufflenet_g1_w1': shufflenet_g1_w1,
    'shufflenet_g2_w1': shufflenet_g2_w1,
    'shufflenet_g3_w1': shufflenet_g3_w1,
    'shufflenet_g4_w1': shufflenet_g4_w1,
    'shufflenet_g8_w1': shufflenet_g8_w1,
    'shufflenet_g1_w3d4': shufflenet_g1_w3d4,
    'shufflenet_g3_w3d4': shufflenet_g3_w3d4,
    'shufflenet_g1_wd2': shufflenet_g1_wd2,
    'shufflenet_g3_wd2': shufflenet_g3_wd2,
    'shufflenet_g1_wd4': shufflenet_g1_wd4,
    'shufflenet_g3_wd4': shufflenet_g3_wd4,

    'shufflenetv2_wd2': shufflenetv2_wd2,
    'shufflenetv2_w1': shufflenetv2_w1,
    'shufflenetv2_w3d2': shufflenetv2_w3d2,
    'shufflenetv2_w2': shufflenetv2_w2,

    'shufflenetv2b_wd2': shufflenetv2b_wd2,
    'shufflenetv2b_w1': shufflenetv2b_w1,
    'shufflenetv2b_w3d2': shufflenetv2b_w3d2,
    'shufflenetv2b_w2': shufflenetv2b_w2,

    'menet108_8x1_g3': menet108_8x1_g3,
    'menet128_8x1_g4': menet128_8x1_g4,
    'menet160_8x1_g8': menet160_8x1_g8,
    'menet228_12x1_g3': menet228_12x1_g3,
    'menet256_12x1_g4': menet256_12x1_g4,
    'menet348_12x1_g3': menet348_12x1_g3,
    'menet352_12x1_g8': menet352_12x1_g8,
    'menet456_24x1_g3': menet456_24x1_g3,

    'mobilenet_w1': mobilenet_w1,
    'mobilenet_w3d4': mobilenet_w3d4,
    'mobilenet_wd2': mobilenet_wd2,
    'mobilenet_wd4': mobilenet_wd4,

    'mobilenetb_w1': mobilenetb_w1,
    'mobilenetb_w3d4': mobilenetb_w3d4,
    'mobilenetb_wd2': mobilenetb_wd2,
    'mobilenetb_wd4': mobilenetb_wd4,

    'fdmobilenet_w1': fdmobilenet_w1,
    'fdmobilenet_w3d4': fdmobilenet_w3d4,
    'fdmobilenet_wd2': fdmobilenet_wd2,
    'fdmobilenet_wd4': fdmobilenet_wd4,

    'mobilenetv2_w1': mobilenetv2_w1,
    'mobilenetv2_w3d4': mobilenetv2_w3d4,
    'mobilenetv2_wd2': mobilenetv2_wd2,
    'mobilenetv2_wd4': mobilenetv2_wd4,
    'mobilenetv2b_w1': mobilenetv2b_w1,
    'mobilenetv2b_w3d4': mobilenetv2b_w3d4,
    'mobilenetv2b_wd2': mobilenetv2b_wd2,
    'mobilenetv2b_wd4': mobilenetv2b_wd4,

    'mobilenetv3_small_w7d20': mobilenetv3_small_w7d20,
    'mobilenetv3_small_wd2': mobilenetv3_small_wd2,
    'mobilenetv3_small_w3d4': mobilenetv3_small_w3d4,
    'mobilenetv3_small_w1': mobilenetv3_small_w1,
    'mobilenetv3_small_w5d4': mobilenetv3_small_w5d4,
    'mobilenetv3_large_w7d20': mobilenetv3_large_w7d20,
    'mobilenetv3_large_wd2': mobilenetv3_large_wd2,
    'mobilenetv3_large_w3d4': mobilenetv3_large_w3d4,
    'mobilenetv3_large_w1': mobilenetv3_large_w1,
    'mobilenetv3_large_w5d4': mobilenetv3_large_w5d4,

    'igcv3_w1': igcv3_w1,
    'igcv3_w3d4': igcv3_w3d4,
    'igcv3_wd2': igcv3_wd2,
    'igcv3_wd4': igcv3_wd4,

    'ghostnet': ghostnet,

    'mnasnet_b1': mnasnet_b1,
    'mnasnet_a1': mnasnet_a1,
    'mnasnet_small': mnasnet_small,

    'darts': darts,

    'proxylessnas_cpu': proxylessnas_cpu,
    'proxylessnas_gpu': proxylessnas_gpu,
    'proxylessnas_mobile': proxylessnas_mobile,
    'proxylessnas_mobile14': proxylessnas_mobile14,

    'fbnet_cb': fbnet_cb,

    'xception': xception,
    'inceptionv3': inceptionv3,
    'inceptionv4': inceptionv4,
    'inceptionresnetv1': inceptionresnetv1,
    'inceptionresnetv2': inceptionresnetv2,
    'polynet': polynet,

    'nasnet_4a1056': nasnet_4a1056,
    'nasnet_6a4032': nasnet_6a4032,

    'pnasnet5large': pnasnet5large,

    'spnasnet': spnasnet,

    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b1': efficientnet_b1,
    'efficientnet_b2': efficientnet_b2,
    'efficientnet_b3': efficientnet_b3,
    'efficientnet_b4': efficientnet_b4,
    'efficientnet_b5': efficientnet_b5,
    'efficientnet_b6': efficientnet_b6,
    'efficientnet_b7': efficientnet_b7,
    'efficientnet_b8': efficientnet_b8,
    'efficientnet_b0b': efficientnet_b0b,
    'efficientnet_b1b': efficientnet_b1b,
    'efficientnet_b2b': efficientnet_b2b,
    'efficientnet_b3b': efficientnet_b3b,
    'efficientnet_b4b': efficientnet_b4b,
    'efficientnet_b5b': efficientnet_b5b,
    'efficientnet_b6b': efficientnet_b6b,
    'efficientnet_b7b': efficientnet_b7b,
    'efficientnet_b0c': efficientnet_b0c,
    'efficientnet_b1c': efficientnet_b1c,
    'efficientnet_b2c': efficientnet_b2c,
    'efficientnet_b3c': efficientnet_b3c,
    'efficientnet_b4c': efficientnet_b4c,
    'efficientnet_b5c': efficientnet_b5c,
    'efficientnet_b6c': efficientnet_b6c,
    'efficientnet_b7c': efficientnet_b7c,
    'efficientnet_b8c': efficientnet_b8c,

    'efficientnet_edge_small_b': efficientnet_edge_small_b,
    'efficientnet_edge_medium_b': efficientnet_edge_medium_b,
    'efficientnet_edge_large_b': efficientnet_edge_large_b,

    'mixnet_s': mixnet_s,
    'mixnet_m': mixnet_m,
    'mixnet_l': mixnet_l,

    'nin_cifar10': nin_cifar10,
    'nin_cifar100': nin_cifar100,
    'nin_svhn': nin_svhn,

    'resnet20_cifar10': resnet20_cifar10,
    'resnet20_cifar100': resnet20_cifar100,
    'resnet20_svhn': resnet20_svhn,
    'resnet56_cifar10': resnet56_cifar10,
    'resnet56_cifar100': resnet56_cifar100,
    'resnet56_svhn': resnet56_svhn,
    'resnet110_cifar10': resnet110_cifar10,
    'resnet110_cifar100': resnet110_cifar100,
    'resnet110_svhn': resnet110_svhn,
    'resnet164bn_cifar10': resnet164bn_cifar10,
    'resnet164bn_cifar100': resnet164bn_cifar100,
    'resnet164bn_svhn': resnet164bn_svhn,
    'resnet272bn_cifar10': resnet272bn_cifar10,
    'resnet272bn_cifar100': resnet272bn_cifar100,
    'resnet272bn_svhn': resnet272bn_svhn,
    'resnet542bn_cifar10': resnet542bn_cifar10,
    'resnet542bn_cifar100': resnet542bn_cifar100,
    'resnet542bn_svhn': resnet542bn_svhn,
    'resnet1001_cifar10': resnet1001_cifar10,
    'resnet1001_cifar100': resnet1001_cifar100,
    'resnet1001_svhn': resnet1001_svhn,
    'resnet1202_cifar10': resnet1202_cifar10,
    'resnet1202_cifar100': resnet1202_cifar100,
    'resnet1202_svhn': resnet1202_svhn,

    'preresnet20_cifar10': preresnet20_cifar10,
    'preresnet20_cifar100': preresnet20_cifar100,
    'preresnet20_svhn': preresnet20_svhn,
    'preresnet56_cifar10': preresnet56_cifar10,
    'preresnet56_cifar100': preresnet56_cifar100,
    'preresnet56_svhn': preresnet56_svhn,
    'preresnet110_cifar10': preresnet110_cifar10,
    'preresnet110_cifar100': preresnet110_cifar100,
    'preresnet110_svhn': preresnet110_svhn,
    'preresnet164bn_cifar10': preresnet164bn_cifar10,
    'preresnet164bn_cifar100': preresnet164bn_cifar100,
    'preresnet164bn_svhn': preresnet164bn_svhn,
    'preresnet272bn_cifar10': preresnet272bn_cifar10,
    'preresnet272bn_cifar100': preresnet272bn_cifar100,
    'preresnet272bn_svhn': preresnet272bn_svhn,
    'preresnet542bn_cifar10': preresnet542bn_cifar10,
    'preresnet542bn_cifar100': preresnet542bn_cifar100,
    'preresnet542bn_svhn': preresnet542bn_svhn,
    'preresnet1001_cifar10': preresnet1001_cifar10,
    'preresnet1001_cifar100': preresnet1001_cifar100,
    'preresnet1001_svhn': preresnet1001_svhn,
    'preresnet1202_cifar10': preresnet1202_cifar10,
    'preresnet1202_cifar100': preresnet1202_cifar100,
    'preresnet1202_svhn': preresnet1202_svhn,

    'resnext20_16x4d_cifar10': resnext20_16x4d_cifar10,
    'resnext20_16x4d_cifar100': resnext20_16x4d_cifar100,
    'resnext20_16x4d_svhn': resnext20_16x4d_svhn,
    'resnext20_32x2d_cifar10': resnext20_32x2d_cifar10,
    'resnext20_32x2d_cifar100': resnext20_32x2d_cifar100,
    'resnext20_32x2d_svhn': resnext20_32x2d_svhn,
    'resnext20_32x4d_cifar10': resnext20_32x4d_cifar10,
    'resnext20_32x4d_cifar100': resnext20_32x4d_cifar100,
    'resnext20_32x4d_svhn': resnext20_32x4d_svhn,
    'resnext29_32x4d_cifar10': resnext29_32x4d_cifar10,
    'resnext29_32x4d_cifar100': resnext29_32x4d_cifar100,
    'resnext29_32x4d_svhn': resnext29_32x4d_svhn,
    'resnext29_16x64d_cifar10': resnext29_16x64d_cifar10,
    'resnext29_16x64d_cifar100': resnext29_16x64d_cifar100,
    'resnext29_16x64d_svhn': resnext29_16x64d_svhn,
    'resnext272_1x64d_cifar10': resnext272_1x64d_cifar10,
    'resnext272_1x64d_cifar100': resnext272_1x64d_cifar100,
    'resnext272_1x64d_svhn': resnext272_1x64d_svhn,
    'resnext272_2x32d_cifar10': resnext272_2x32d_cifar10,
    'resnext272_2x32d_cifar100': resnext272_2x32d_cifar100,
    'resnext272_2x32d_svhn': resnext272_2x32d_svhn,

    'seresnet20_cifar10': seresnet20_cifar10,
    'seresnet20_cifar100': seresnet20_cifar100,
    'seresnet20_svhn': seresnet20_svhn,
    'seresnet56_cifar10': seresnet56_cifar10,
    'seresnet56_cifar100': seresnet56_cifar100,
    'seresnet56_svhn': seresnet56_svhn,
    'seresnet110_cifar10': seresnet110_cifar10,
    'seresnet110_cifar100': seresnet110_cifar100,
    'seresnet110_svhn': seresnet110_svhn,
    'seresnet164bn_cifar10': seresnet164bn_cifar10,
    'seresnet164bn_cifar100': seresnet164bn_cifar100,
    'seresnet164bn_svhn': seresnet164bn_svhn,
    'seresnet272bn_cifar10': seresnet272bn_cifar10,
    'seresnet272bn_cifar100': seresnet272bn_cifar100,
    'seresnet272bn_svhn': seresnet272bn_svhn,
    'seresnet542bn_cifar10': seresnet542bn_cifar10,
    'seresnet542bn_cifar100': seresnet542bn_cifar100,
    'seresnet542bn_svhn': seresnet542bn_svhn,
    'seresnet1001_cifar10': seresnet1001_cifar10,
    'seresnet1001_cifar100': seresnet1001_cifar100,
    'seresnet1001_svhn': seresnet1001_svhn,
    'seresnet1202_cifar10': seresnet1202_cifar10,
    'seresnet1202_cifar100': seresnet1202_cifar100,
    'seresnet1202_svhn': seresnet1202_svhn,

    'sepreresnet20_cifar10': sepreresnet20_cifar10,
    'sepreresnet20_cifar100': sepreresnet20_cifar100,
    'sepreresnet20_svhn': sepreresnet20_svhn,
    'sepreresnet56_cifar10': sepreresnet56_cifar10,
    'sepreresnet56_cifar100': sepreresnet56_cifar100,
    'sepreresnet56_svhn': sepreresnet56_svhn,
    'sepreresnet110_cifar10': sepreresnet110_cifar10,
    'sepreresnet110_cifar100': sepreresnet110_cifar100,
    'sepreresnet110_svhn': sepreresnet110_svhn,
    'sepreresnet164bn_cifar10': sepreresnet164bn_cifar10,
    'sepreresnet164bn_cifar100': sepreresnet164bn_cifar100,
    'sepreresnet164bn_svhn': sepreresnet164bn_svhn,
    'sepreresnet272bn_cifar10': sepreresnet272bn_cifar10,
    'sepreresnet272bn_cifar100': sepreresnet272bn_cifar100,
    'sepreresnet272bn_svhn': sepreresnet272bn_svhn,
    'sepreresnet542bn_cifar10': sepreresnet542bn_cifar10,
    'sepreresnet542bn_cifar100': sepreresnet542bn_cifar100,
    'sepreresnet542bn_svhn': sepreresnet542bn_svhn,
    'sepreresnet1001_cifar10': sepreresnet1001_cifar10,
    'sepreresnet1001_cifar100': sepreresnet1001_cifar100,
    'sepreresnet1001_svhn': sepreresnet1001_svhn,
    'sepreresnet1202_cifar10': sepreresnet1202_cifar10,
    'sepreresnet1202_cifar100': sepreresnet1202_cifar100,
    'sepreresnet1202_svhn': sepreresnet1202_svhn,

    'pyramidnet110_a48_cifar10': pyramidnet110_a48_cifar10,
    'pyramidnet110_a48_cifar100': pyramidnet110_a48_cifar100,
    'pyramidnet110_a48_svhn': pyramidnet110_a48_svhn,
    'pyramidnet110_a84_cifar10': pyramidnet110_a84_cifar10,
    'pyramidnet110_a84_cifar100': pyramidnet110_a84_cifar100,
    'pyramidnet110_a84_svhn': pyramidnet110_a84_svhn,
    'pyramidnet110_a270_cifar10': pyramidnet110_a270_cifar10,
    'pyramidnet110_a270_cifar100': pyramidnet110_a270_cifar100,
    'pyramidnet110_a270_svhn': pyramidnet110_a270_svhn,
    'pyramidnet164_a270_bn_cifar10': pyramidnet164_a270_bn_cifar10,
    'pyramidnet164_a270_bn_cifar100': pyramidnet164_a270_bn_cifar100,
    'pyramidnet164_a270_bn_svhn': pyramidnet164_a270_bn_svhn,
    'pyramidnet200_a240_bn_cifar10': pyramidnet200_a240_bn_cifar10,
    'pyramidnet200_a240_bn_cifar100': pyramidnet200_a240_bn_cifar100,
    'pyramidnet200_a240_bn_svhn': pyramidnet200_a240_bn_svhn,
    'pyramidnet236_a220_bn_cifar10': pyramidnet236_a220_bn_cifar10,
    'pyramidnet236_a220_bn_cifar100': pyramidnet236_a220_bn_cifar100,
    'pyramidnet236_a220_bn_svhn': pyramidnet236_a220_bn_svhn,
    'pyramidnet272_a200_bn_cifar10': pyramidnet272_a200_bn_cifar10,
    'pyramidnet272_a200_bn_cifar100': pyramidnet272_a200_bn_cifar100,
    'pyramidnet272_a200_bn_svhn': pyramidnet272_a200_bn_svhn,

    'densenet40_k12_cifar10': densenet40_k12_cifar10,
    'densenet40_k12_cifar100': densenet40_k12_cifar100,
    'densenet40_k12_svhn': densenet40_k12_svhn,
    'densenet40_k12_bc_cifar10': densenet40_k12_bc_cifar10,
    'densenet40_k12_bc_cifar100': densenet40_k12_bc_cifar100,
    'densenet40_k12_bc_svhn': densenet40_k12_bc_svhn,
    'densenet40_k24_bc_cifar10': densenet40_k24_bc_cifar10,
    'densenet40_k24_bc_cifar100': densenet40_k24_bc_cifar100,
    'densenet40_k24_bc_svhn': densenet40_k24_bc_svhn,
    'densenet40_k36_bc_cifar10': densenet40_k36_bc_cifar10,
    'densenet40_k36_bc_cifar100': densenet40_k36_bc_cifar100,
    'densenet40_k36_bc_svhn': densenet40_k36_bc_svhn,
    'densenet100_k12_cifar10': densenet100_k12_cifar10,
    'densenet100_k12_cifar100': densenet100_k12_cifar100,
    'densenet100_k12_svhn': densenet100_k12_svhn,
    'densenet100_k24_cifar10': densenet100_k24_cifar10,
    'densenet100_k24_cifar100': densenet100_k24_cifar100,
    'densenet100_k24_svhn': densenet100_k24_svhn,
    'densenet100_k12_bc_cifar10': densenet100_k12_bc_cifar10,
    'densenet100_k12_bc_cifar100': densenet100_k12_bc_cifar100,
    'densenet100_k12_bc_svhn': densenet100_k12_bc_svhn,
    'densenet190_k40_bc_cifar10': densenet190_k40_bc_cifar10,
    'densenet190_k40_bc_cifar100': densenet190_k40_bc_cifar100,
    'densenet190_k40_bc_svhn': densenet190_k40_bc_svhn,
    'densenet250_k24_bc_cifar10': densenet250_k24_bc_cifar10,
    'densenet250_k24_bc_cifar100': densenet250_k24_bc_cifar100,
    'densenet250_k24_bc_svhn': densenet250_k24_bc_svhn,

    'xdensenet40_2_k24_bc_cifar10': xdensenet40_2_k24_bc_cifar10,
    'xdensenet40_2_k24_bc_cifar100': xdensenet40_2_k24_bc_cifar100,
    'xdensenet40_2_k24_bc_svhn': xdensenet40_2_k24_bc_svhn,
    'xdensenet40_2_k36_bc_cifar10': xdensenet40_2_k36_bc_cifar10,
    'xdensenet40_2_k36_bc_cifar100': xdensenet40_2_k36_bc_cifar100,
    'xdensenet40_2_k36_bc_svhn': xdensenet40_2_k36_bc_svhn,

    'wrn16_10_cifar10': wrn16_10_cifar10,
    'wrn16_10_cifar100': wrn16_10_cifar100,
    'wrn16_10_svhn': wrn16_10_svhn,
    'wrn28_10_cifar10': wrn28_10_cifar10,
    'wrn28_10_cifar100': wrn28_10_cifar100,
    'wrn28_10_svhn': wrn28_10_svhn,
    'wrn40_8_cifar10': wrn40_8_cifar10,
    'wrn40_8_cifar100': wrn40_8_cifar100,
    'wrn40_8_svhn': wrn40_8_svhn,

    'wrn20_10_1bit_cifar10': wrn20_10_1bit_cifar10,
    'wrn20_10_1bit_cifar100': wrn20_10_1bit_cifar100,
    'wrn20_10_1bit_svhn': wrn20_10_1bit_svhn,
    'wrn20_10_32bit_cifar10': wrn20_10_32bit_cifar10,
    'wrn20_10_32bit_cifar100': wrn20_10_32bit_cifar100,
    'wrn20_10_32bit_svhn': wrn20_10_32bit_svhn,

    'ror3_56_cifar10': ror3_56_cifar10,
    'ror3_56_cifar100': ror3_56_cifar100,
    'ror3_56_svhn': ror3_56_svhn,
    'ror3_110_cifar10': ror3_110_cifar10,
    'ror3_110_cifar100': ror3_110_cifar100,
    'ror3_110_svhn': ror3_110_svhn,
    'ror3_164_cifar10': ror3_164_cifar10,
    'ror3_164_cifar100': ror3_164_cifar100,
    'ror3_164_svhn': ror3_164_svhn,

    'rir_cifar10': rir_cifar10,
    'rir_cifar100': rir_cifar100,
    'rir_svhn': rir_svhn,

    'msdnet22_cifar10': msdnet22_cifar10,

    'resdropresnet20_cifar10': resdropresnet20_cifar10,
    'resdropresnet20_cifar100': resdropresnet20_cifar100,
    'resdropresnet20_svhn': resdropresnet20_svhn,

    'shakeshakeresnet20_2x16d_cifar10': shakeshakeresnet20_2x16d_cifar10,
    'shakeshakeresnet20_2x16d_cifar100': shakeshakeresnet20_2x16d_cifar100,
    'shakeshakeresnet20_2x16d_svhn': shakeshakeresnet20_2x16d_svhn,
    'shakeshakeresnet26_2x32d_cifar10': shakeshakeresnet26_2x32d_cifar10,
    'shakeshakeresnet26_2x32d_cifar100': shakeshakeresnet26_2x32d_cifar100,
    'shakeshakeresnet26_2x32d_svhn': shakeshakeresnet26_2x32d_svhn,

    'shakedropresnet20_cifar10': shakedropresnet20_cifar10,
    'shakedropresnet20_cifar100': shakedropresnet20_cifar100,
    'shakedropresnet20_svhn': shakedropresnet20_svhn,

    'fractalnet_cifar10': fractalnet_cifar10,
    'fractalnet_cifar100': fractalnet_cifar100,

    'diaresnet20_cifar10': diaresnet20_cifar10,
    'diaresnet20_cifar100': diaresnet20_cifar100,
    'diaresnet20_svhn': diaresnet20_svhn,
    'diaresnet56_cifar10': diaresnet56_cifar10,
    'diaresnet56_cifar100': diaresnet56_cifar100,
    'diaresnet56_svhn': diaresnet56_svhn,
    'diaresnet110_cifar10': diaresnet110_cifar10,
    'diaresnet110_cifar100': diaresnet110_cifar100,
    'diaresnet110_svhn': diaresnet110_svhn,
    'diaresnet164bn_cifar10': diaresnet164bn_cifar10,
    'diaresnet164bn_cifar100': diaresnet164bn_cifar100,
    'diaresnet164bn_svhn': diaresnet164bn_svhn,
    'diaresnet1001_cifar10': diaresnet1001_cifar10,
    'diaresnet1001_cifar100': diaresnet1001_cifar100,
    'diaresnet1001_svhn': diaresnet1001_svhn,
    'diaresnet1202_cifar10': diaresnet1202_cifar10,
    'diaresnet1202_cifar100': diaresnet1202_cifar100,
    'diaresnet1202_svhn': diaresnet1202_svhn,

    'diapreresnet20_cifar10': diapreresnet20_cifar10,
    'diapreresnet20_cifar100': diapreresnet20_cifar100,
    'diapreresnet20_svhn': diapreresnet20_svhn,
    'diapreresnet56_cifar10': diapreresnet56_cifar10,
    'diapreresnet56_cifar100': diapreresnet56_cifar100,
    'diapreresnet56_svhn': diapreresnet56_svhn,
    'diapreresnet110_cifar10': diapreresnet110_cifar10,
    'diapreresnet110_cifar100': diapreresnet110_cifar100,
    'diapreresnet110_svhn': diapreresnet110_svhn,
    'diapreresnet164bn_cifar10': diapreresnet164bn_cifar10,
    'diapreresnet164bn_cifar100': diapreresnet164bn_cifar100,
    'diapreresnet164bn_svhn': diapreresnet164bn_svhn,
    'diapreresnet1001_cifar10': diapreresnet1001_cifar10,
    'diapreresnet1001_cifar100': diapreresnet1001_cifar100,
    'diapreresnet1001_svhn': diapreresnet1001_svhn,
    'diapreresnet1202_cifar10': diapreresnet1202_cifar10,
    'diapreresnet1202_cifar100': diapreresnet1202_cifar100,
    'diapreresnet1202_svhn': diapreresnet1202_svhn,

    'isqrtcovresnet18': isqrtcovresnet18,
    'isqrtcovresnet34': isqrtcovresnet34,
    'isqrtcovresnet50': isqrtcovresnet50,
    'isqrtcovresnet50b': isqrtcovresnet50b,
    'isqrtcovresnet101': isqrtcovresnet101,
    'isqrtcovresnet101b': isqrtcovresnet101b,

    'resneta10': resneta10,
    'resnetabc14b': resnetabc14b,
    'resneta18': resneta18,
    'resneta50b': resneta50b,
    'resneta101b': resneta101b,
    'resneta152b': resneta152b,

    'resnetd50b': resnetd50b,
    'resnetd101b': resnetd101b,
    'resnetd152b': resnetd152b,

    'fastseresnet101b': fastseresnet101b,

    'octresnet10_ad2': octresnet10_ad2,
    'octresnet50b_ad2': octresnet50b_ad2,

    'resnet10_cub': resnet10_cub,
    'resnet12_cub': resnet12_cub,
    'resnet14_cub': resnet14_cub,
    'resnetbc14b_cub': resnetbc14b_cub,
    'resnet16_cub': resnet16_cub,
    'resnet18_cub': resnet18_cub,
    'resnet26_cub': resnet26_cub,
    'resnetbc26b_cub': resnetbc26b_cub,
    'resnet34_cub': resnet34_cub,
    'resnetbc38b_cub': resnetbc38b_cub,
    'resnet50_cub': resnet50_cub,
    'resnet50b_cub': resnet50b_cub,
    'resnet101_cub': resnet101_cub,
    'resnet101b_cub': resnet101b_cub,
    'resnet152_cub': resnet152_cub,
    'resnet152b_cub': resnet152b_cub,
    'resnet200_cub': resnet200_cub,
    'resnet200b_cub': resnet200b_cub,

    'seresnet10_cub': seresnet10_cub,
    'seresnet12_cub': seresnet12_cub,
    'seresnet14_cub': seresnet14_cub,
    'seresnetbc14b_cub': seresnetbc14b_cub,
    'seresnet16_cub': seresnet16_cub,
    'seresnet18_cub': seresnet18_cub,
    'seresnet26_cub': seresnet26_cub,
    'seresnetbc26b_cub': seresnetbc26b_cub,
    'seresnet34_cub': seresnet34_cub,
    'seresnetbc38b_cub': seresnetbc38b_cub,
    'seresnet50_cub': seresnet50_cub,
    'seresnet50b_cub': seresnet50b_cub,
    'seresnet101_cub': seresnet101_cub,
    'seresnet101b_cub': seresnet101b_cub,
    'seresnet152_cub': seresnet152_cub,
    'seresnet152b_cub': seresnet152b_cub,
    'seresnet200_cub': seresnet200_cub,
    'seresnet200b_cub': seresnet200b_cub,

    'mobilenet_w1_cub': mobilenet_w1_cub,
    'mobilenet_w3d4_cub': mobilenet_w3d4_cub,
    'mobilenet_wd2_cub': mobilenet_wd2_cub,
    'mobilenet_wd4_cub': mobilenet_wd4_cub,

    'fdmobilenet_w1_cub': fdmobilenet_w1_cub,
    'fdmobilenet_w3d4_cub': fdmobilenet_w3d4_cub,
    'fdmobilenet_wd2_cub': fdmobilenet_wd2_cub,
    'fdmobilenet_wd4_cub': fdmobilenet_wd4_cub,

    'proxylessnas_cpu_cub': proxylessnas_cpu_cub,
    'proxylessnas_gpu_cub': proxylessnas_gpu_cub,
    'proxylessnas_mobile_cub': proxylessnas_mobile_cub,
    'proxylessnas_mobile14_cub': proxylessnas_mobile14_cub,

    'ntsnet_cub': ntsnet_cub,

    'fcn8sd_resnetd50b_voc': fcn8sd_resnetd50b_voc,
    'fcn8sd_resnetd101b_voc': fcn8sd_resnetd101b_voc,
    'fcn8sd_resnetd50b_coco': fcn8sd_resnetd50b_coco,
    'fcn8sd_resnetd101b_coco': fcn8sd_resnetd101b_coco,
    'fcn8sd_resnetd50b_ade20k': fcn8sd_resnetd50b_ade20k,
    'fcn8sd_resnetd101b_ade20k': fcn8sd_resnetd101b_ade20k,
    'fcn8sd_resnetd50b_cityscapes': fcn8sd_resnetd50b_cityscapes,
    'fcn8sd_resnetd101b_cityscapes': fcn8sd_resnetd101b_cityscapes,

    'pspnet_resnetd50b_voc': pspnet_resnetd50b_voc,
    'pspnet_resnetd101b_voc': pspnet_resnetd101b_voc,
    'pspnet_resnetd50b_coco': pspnet_resnetd50b_coco,
    'pspnet_resnetd101b_coco': pspnet_resnetd101b_coco,
    'pspnet_resnetd50b_ade20k': pspnet_resnetd50b_ade20k,
    'pspnet_resnetd101b_ade20k': pspnet_resnetd101b_ade20k,
    'pspnet_resnetd50b_cityscapes': pspnet_resnetd50b_cityscapes,
    'pspnet_resnetd101b_cityscapes': pspnet_resnetd101b_cityscapes,

    'deeplabv3_resnetd50b_voc': deeplabv3_resnetd50b_voc,
    'deeplabv3_resnetd101b_voc': deeplabv3_resnetd101b_voc,
    'deeplabv3_resnetd152b_voc': deeplabv3_resnetd152b_voc,
    'deeplabv3_resnetd50b_coco': deeplabv3_resnetd50b_coco,
    'deeplabv3_resnetd101b_coco': deeplabv3_resnetd101b_coco,
    'deeplabv3_resnetd152b_coco': deeplabv3_resnetd152b_coco,
    'deeplabv3_resnetd50b_ade20k': deeplabv3_resnetd50b_ade20k,
    'deeplabv3_resnetd101b_ade20k': deeplabv3_resnetd101b_ade20k,
    'deeplabv3_resnetd50b_cityscapes': deeplabv3_resnetd50b_cityscapes,
    'deeplabv3_resnetd101b_cityscapes': deeplabv3_resnetd101b_cityscapes,

    'icnet_resnetd50b_cityscapes': icnet_resnetd50b_cityscapes,

    'fastscnn_cityscapes': fastscnn_cityscapes,

    'cgnet_cityscapes': cgnet_cityscapes,
    'dabnet_cityscapes': dabnet_cityscapes,

    'sinet_cityscapes': sinet_cityscapes,

    'bisenet_resnet18_celebamaskhq': bisenet_resnet18_celebamaskhq,

    'danet_resnetd50b_cityscapes': danet_resnetd50b_cityscapes,
    'danet_resnetd101b_cityscapes': danet_resnetd101b_cityscapes,

    'fpenet_cityscapes': fpenet_cityscapes,

    'ctxnet_cityscapes': ctxnet_cityscapes,

    'lednet_cityscapes': lednet_cityscapes,

    'esnet_cityscapes': esnet_cityscapes,

    'edanet_cityscapes': edanet_cityscapes,

    'enet_cityscapes': enet_cityscapes,

    'erfnet_cityscapes': erfnet_cityscapes,

    'linknet_cityscapes': linknet_cityscapes,

    'segnet_cityscapes': segnet_cityscapes,

    'unet_cityscapes': unet_cityscapes,

    'sqnet_cityscapes': sqnet_cityscapes,

    'alphapose_fastseresnet101b_coco': alphapose_fastseresnet101b_coco,

    'simplepose_resnet18_coco': simplepose_resnet18_coco,
    'simplepose_resnet50b_coco': simplepose_resnet50b_coco,
    'simplepose_resnet101b_coco': simplepose_resnet101b_coco,
    'simplepose_resnet152b_coco': simplepose_resnet152b_coco,
    'simplepose_resneta50b_coco': simplepose_resneta50b_coco,
    'simplepose_resneta101b_coco': simplepose_resneta101b_coco,
    'simplepose_resneta152b_coco': simplepose_resneta152b_coco,

    'simplepose_mobile_resnet18_coco': simplepose_mobile_resnet18_coco,
    'simplepose_mobile_resnet50b_coco': simplepose_mobile_resnet50b_coco,
    'simplepose_mobile_mobilenet_w1_coco': simplepose_mobile_mobilenet_w1_coco,
    'simplepose_mobile_mobilenetv2b_w1_coco': simplepose_mobile_mobilenetv2b_w1_coco,
    'simplepose_mobile_mobilenetv3_small_w1_coco': simplepose_mobile_mobilenetv3_small_w1_coco,
    'simplepose_mobile_mobilenetv3_large_w1_coco': simplepose_mobile_mobilenetv3_large_w1_coco,

    'lwopenpose2d_mobilenet_cmupan_coco': lwopenpose2d_mobilenet_cmupan_coco,
    'lwopenpose3d_mobilenet_cmupan_coco': lwopenpose3d_mobilenet_cmupan_coco,

    'ibppose_coco': ibppose_coco,

    'prnet': prnet,

    'centernet_resnet18_voc': centernet_resnet18_voc,
    'centernet_resnet18_coco': centernet_resnet18_coco,
    'centernet_resnet50b_voc': centernet_resnet50b_voc,
    'centernet_resnet50b_coco': centernet_resnet50b_coco,
    'centernet_resnet101b_voc': centernet_resnet101b_voc,
    'centernet_resnet101b_coco': centernet_resnet101b_coco,

    'lffd20x5s320v2_widerface': lffd20x5s320v2_widerface,
    'lffd25x8s560v1_widerface': lffd25x8s560v1_widerface,

    'pfpcnet': pfpcnet,
    'voca8flame': voca8flame,
    'nvpattexp116bazel76': nvpattexp116bazel76,
    "visemenet20": visemenet20,

    'superpointnet': superpointnet,

    'jasper5x3': jasper5x3,
    'jasper10x4': jasper10x4,
    'jasper10x5': jasper10x5,

    'jasperdr10x5_en': jasperdr10x5_en,
    'jasperdr10x5_en_nr': jasperdr10x5_en_nr,

    'quartznet5x5_en_ls': quartznet5x5_en_ls,
    'quartznet15x5_en': quartznet15x5_en,
    'quartznet15x5_en_nr': quartznet15x5_en_nr,
    'quartznet15x5_fr': quartznet15x5_fr,
    'quartznet15x5_de': quartznet15x5_de,
    'quartznet15x5_it': quartznet15x5_it,
    'quartznet15x5_es': quartznet15x5_es,
    'quartznet15x5_ca': quartznet15x5_ca,
    'quartznet15x5_pl': quartznet15x5_pl,
    'quartznet15x5_ru': quartznet15x5_ru,
    'quartznet15x5_ru34': quartznet15x5_ru34,

    'tresnet_m': tresnet_m,
    'tresnet_l': tresnet_l,
    'tresnet_xl': tresnet_xl,

    "espcnet_cityscapes": espcnet_cityscapes,

    'raft_things': raft_things,
    'raft_small': raft_small,

    'propainter_rfc': propainter_rfc,
    'propainter_ip': propainter_ip,
    'propainter': propainter,

    # 'oth_quartznet5x5_en_ls': oth_quartznet5x5_en_ls,
    # 'oth_quartznet15x5_en': oth_quartznet15x5_en,
    # 'oth_quartznet15x5_en_nr': oth_quartznet15x5_en_nr,
    # 'oth_quartznet15x5_fr': oth_quartznet15x5_fr,
    # 'oth_quartznet15x5_de': oth_quartznet15x5_de,
    # 'oth_quartznet15x5_it': oth_quartznet15x5_it,
    # 'oth_quartznet15x5_es': oth_quartznet15x5_es,
    # 'oth_quartznet15x5_ca': oth_quartznet15x5_ca,
    # 'oth_quartznet15x5_pl': oth_quartznet15x5_pl,
    # 'oth_quartznet15x5_ru': oth_quartznet15x5_ru,
    # 'oth_jasperdr10x5_en': oth_jasperdr10x5_en,
    # 'oth_jasperdr10x5_en_nr': oth_jasperdr10x5_en_nr,
    # 'oth_quartznet15x5_ru34': oth_quartznet15x5_ru34,

    # 'oth_pose_coco_resnet_50_256x192': oth_pose_coco_resnet_50_256x192,
    # 'oth_pose_coco_resnet_50_384x288': oth_pose_coco_resnet_50_384x288,
    # 'oth_pose_coco_resnet_101_256x192': oth_pose_coco_resnet_101_256x192,
    # 'oth_pose_coco_resnet_101_384x288': oth_pose_coco_resnet_101_384x288,
    # 'oth_pose_coco_resnet_152_256x192': oth_pose_coco_resnet_152_256x192,
    # 'oth_pose_coco_resnet_152_384x288': oth_pose_coco_resnet_152_384x288,

    # 'oth_lwopenpose2d': oth_lwopenpose2d,
    # 'oth_lwopenpose3d': oth_lwopenpose3d,

    # 'oth_prnet': oth_prnet,

    # 'oth_sinet_cityscapes': oth_sinet_cityscapes,

    # 'oth_ibppose': oth_ibppose,

    # 'oth_bisenet': oth_bisenet,

    # 'oth_tresnet_m': oth_tresnet_m,

    # 'oth_dabnet_cityscapes': oth_dabnet_cityscapes,
}


def get_model(name, **kwargs):
    """
    Get supported model.

    Parameters
    ----------
    name : str
        Name of model.

    Returns
    -------
    Module
        Resulted model.
    """
    name = name.lower()
    if name not in _models:
        raise ValueError("Unsupported model: {}".format(name))
    net = _models[name](**kwargs)
    return net
