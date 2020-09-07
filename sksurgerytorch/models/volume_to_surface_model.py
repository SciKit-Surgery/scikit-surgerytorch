
""" Model implemetnation for V2SNet"""
import time
import torchvision
import numpy
import torch
import torch.nn as nn


#pylint:disable=invalid-name, missing-function-docstring
#pylint:disable=missing-class-docstring, redefined-outer-name
#pylint:disable=unused-variable, super-with-arguments, line-too-long
#pylint:disable=attribute-defined-outside-init, too-many-instance-attributes
#pylint:disable=access-member-before-definition

class ConstPaddedConv(nn.Module):
    def __init__(
            self,
            cIn,
            cOut,
            kernel_size,
            stride=1,
            dilation=1,
            padding=0):
        super(ConstPaddedConv, self).__init__()
        #self.padder = nn.ConstantPad3d( padding, -10 )
        self.conv = nn.Conv3d(
            cIn,
            cOut,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0)
        self.padding = padding

    def forward(self, data):
        if self.padding > 0:
            padding = [self.padding] * 6
            data = nn.functional.pad(data, padding, mode="replicate")
        return self.conv(data)


class MultiDilationUnit(nn.Module):

    def __init__(
            self,
            channelsIn,
            channelsDilated,
            channelsOut,
            baseKernelSize=7):
        super(MultiDilationUnit, self).__init__()

        self.preConv = ConstPaddedConv(
            channelsIn, channelsDilated, kernel_size=3, padding=1)

        self.dilationModules = torch.nn.ModuleList()
        dilations = [1, baseKernelSize - 2]

        numChannels = 0
        for dilation in dilations:
            for kernelSize in [baseKernelSize]:
                p = int(((kernelSize - 1) / 2) * dilation)
                conv = ConstPaddedConv(
                    channelsDilated,
                    channelsDilated,
                    kernel_size=kernelSize,
                    dilation=dilation,
                    padding=p)
                self.dilationModules.append(conv)
                numChannels += channelsDilated

        self.combineConv = ConstPaddedConv(
            numChannels, channelsOut, kernel_size=1, padding=0)

        self.nonLin = nn.Softsign()

    def forward(self, data):

        data = self.nonLin(self.preConv(data))
        d = []
        for i, conv in enumerate(self.dilationModules):
            dilation = conv(data)
            d.append(dilation)
            #d.append( self.nonLin( dilation ) )

        dilations = torch.cat(d, dim=1)

        res = self.nonLin(self.combineConv(dilations))
        return res


class Model(nn.Module):

    def __init__(self, mask=True):
        super(Model, self).__init__()

        self.mask = mask

        self.conv64_0 = ConstPaddedConv(
            2, 16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv64_1 = ConstPaddedConv(
            16, 24, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv64_2 = ConstPaddedConv(
            24, 24, kernel_size=3, stride=1, padding=1, dilation=1)

        self.down64to32 = ConstPaddedConv(
            24, 32, kernel_size=4, stride=2, padding=1)

        self.conv32_0 = ConstPaddedConv(
            32, 32, kernel_size=3, stride=1, padding=1)
        self.conv32_1 = ConstPaddedConv(
            32, 32, kernel_size=3, stride=1, padding=1)
        self.conv32_2 = ConstPaddedConv(
            32, 32, kernel_size=3, stride=1, padding=1)

        self.down32to16 = ConstPaddedConv(
            32, 64, kernel_size=4, stride=2, padding=1)

        self.conv16_0 = ConstPaddedConv(
            64, 64, kernel_size=3, stride=1, padding=1)
        self.conv16_1 = ConstPaddedConv(
            64, 64, kernel_size=3, stride=1, padding=1)
        self.conv16_2 = ConstPaddedConv(
            64, 64, kernel_size=3, stride=1, padding=1)

        self.down16to8 = ConstPaddedConv(
            64, 96, kernel_size=4, stride=2, padding=1)

        self.conv8_0 = ConstPaddedConv(
            96, 96, kernel_size=3, stride=1, padding=1)
        self.conv8_1 = ConstPaddedConv(
            96, 96, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = ConstPaddedConv(
            96, 96, kernel_size=3, stride=1, padding=1)
        self.conv8_3 = ConstPaddedConv(
            96, 96, kernel_size=3, stride=1, padding=1)
        self.conv8_4 = ConstPaddedConv(
            96, 96, kernel_size=3, stride=1, padding=1)
        self.conv8_5 = ConstPaddedConv(
            96, 96, kernel_size=3, stride=1, padding=1)

        self.conv16_combine_0 = ConstPaddedConv(
            96 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv16_combine_1 = ConstPaddedConv(
            64, 64, kernel_size=3, stride=1, padding=1)

        self.conv32_combine_0 = ConstPaddedConv(
            64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv32_combine_1 = ConstPaddedConv(
            32, 32, kernel_size=3, stride=1, padding=1)

        self.conv64_combine_0 = ConstPaddedConv(
            32 + 24, 24, kernel_size=3, stride=1, padding=1)
        self.conv64_combine_1 = ConstPaddedConv(
            24, 16, kernel_size=3, stride=1, padding=1)
        self.conv64_combine_2 = ConstPaddedConv(
            16, 3, kernel_size=3, stride=1, padding=1)

        self.conv32toOutput = ConstPaddedConv(32, 3, kernel_size=1)
        self.conv16toOutput = ConstPaddedConv(64, 3, kernel_size=1)
        self.conv8toOutput = ConstPaddedConv(96, 3, kernel_size=1)

        self.nonLin = nn.Softsign()

        self.timing = False

    def time(self, label=None):
        if self.timing:
            torch.cuda.synchronize()
            if label is not None:
                print(label, time.perf_counter() - self.dt)
            self.dt = time.perf_counter()

    def forward(self, preoperative, intraoperative):

        config = torch.cat((preoperative, intraoperative), 1)

        self.time()

        res64 = self.nonLin(self.conv64_0(config))
        res64 = self.nonLin(self.conv64_1(res64))
        res64 = self.nonLin(self.conv64_2(res64))

        self.time(0)

        res32 = self.nonLin(self.down64to32(res64))
        self.time(1)

        res32 = self.nonLin(self.conv32_0(res32))
        res32 = self.nonLin(self.conv32_1(res32))
        res32 = self.nonLin(self.conv32_2(res32))
        self.time(2)

        res16 = self.nonLin(self.down32to16(res32))
        self.time(3)

        res16 = self.nonLin(self.conv16_0(res16))
        res16 = self.nonLin(self.conv16_1(res16))
        res16 = self.nonLin(self.conv16_2(res16))
        self.time(4)

        res8 = self.nonLin(self.down16to8(res16))
        self.time(5)

        res8 = self.nonLin(self.conv8_0(res8))
        res8 = self.nonLin(self.conv8_1(res8))
        res8 = self.nonLin(self.conv8_2(res8))
        res8 = self.nonLin(self.conv8_3(res8))
        res8 = self.nonLin(self.conv8_4(res8))
        res8 = self.nonLin(self.conv8_5(res8))
        self.time(6)

        up16 = torch.nn.functional.interpolate(res8, size=16)
        self.time(7)

        res16 = torch.cat((up16, res16), dim=1)
        res16 = self.nonLin(self.conv16_combine_0(res16))
        res16 = self.nonLin(self.conv16_combine_1(res16))
        self.time(8)

        up32 = torch.nn.functional.interpolate(res16, size=32)
        self.time(9)

        res32 = torch.cat((up32, res32), dim=1)
        res32 = self.nonLin(self.conv32_combine_0(res32))
        res32 = self.nonLin(self.conv32_combine_1(res32))
        self.time(10)

        up64 = torch.nn.functional.interpolate(res32, size=64)

        self.time(11)
        res64 = torch.cat((up64, res64), dim=1)
        res64 = self.nonLin(self.conv64_combine_0(res64))
        res64 = self.nonLin(self.conv64_combine_1(res64))
        res64 = self.nonLin(self.conv64_combine_2(res64))
        self.time(12)

        res64out = res64
        self.time(13)

        # Lower resolution outputs for additional error terms:
        res32out = self.nonLin(self.conv32toOutput(res32))
        res16out = self.nonLin(self.conv16toOutput(res16))
        res8out = self.nonLin(self.conv8toOutput(res8))
        self.time(14)

        # Generate mask from signed distance function:
        mask = preoperative.lt(0)
        mask = mask.expand(-1, 3, -1, -1, -1).float()

        if self.mask:
            res64out = res64out * mask
            res32out = res32out * \
                torch.nn.functional.interpolate(mask, size=32)
            res16out = res16out * \
                torch.nn.functional.interpolate(mask, size=16)
            res8out = res8out * torch.nn.functional.interpolate(mask, size=8)

        self.time(15)
        return res64out, res32out, res16out, res8out

    def init(self):
        self.apply(initWeightsAverage)


class TestModel(nn.Module):

    def __init__(self, mask=True):
        super(TestModel, self).__init__()

        self.mask = mask

        self.conv0 = ConstPaddedConv(2, 1, kernel_size=3, stride=1, padding=1)
        self.convs = nn.ModuleList()
        for i in range(100):
            self.convs.append(
                ConstPaddedConv(
                    1,
                    1,
                    kernel_size=3,
                    stride=1,
                    padding=1))
        self.convX = ConstPaddedConv(1, 3, kernel_size=3, stride=1, padding=1)

        self.nonLin = nn.Softsign()

        self.timing = False

    def time(self, label=None):
        if self.timing:
            torch.cuda.synchronize()
            if label is not None:
                print(label, time.perf_counter() - self.dt)
            self.dt = time.perf_counter()

    def forward(self, preoperative, intraoperative):

        config = torch.cat((preoperative, intraoperative), 1)

        res64 = self.nonLin(self.conv0(config))
        for c in self.convs:
            res64 = self.nonLin(c(res64))
        res64out = self.nonLin(self.convX(res64))

        res32out = torch.nn.functional.interpolate(res64out, size=32)
        res16out = torch.nn.functional.interpolate(res64out, size=16)
        res8out = torch.nn.functional.interpolate(res64out, size=8)

        self.time(15)
        return res64out, res32out, res16out, res8out

# Initialize to do an "average" operation only:


def initWeightsAverage(m):
    if isinstance(m, nn.Conv3d):
        m.bias.data = torch.zeros_like(m.bias)

        # Calculate number of input channels times kernel size:
        s = m.weight.shape
        numInputs = s[1] * s[2] * s[3] * s[4]
        m.weight.data = torch.ones_like(m.weight) / numInputs
        m.weight.data += torch.randn_like(m.weight) / numInputs


def initWeights(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_normal_(m.weight.data)
        m.bias.data.zero_()


# Perform an occlusion experiment to see which inputs influence a final
# output pixel:
def occlusionExperiment(
        model,
        preoperative,
        intraoperative,
        occ_size=4,
        occ_stride=4,
        occ_pixel=1):

    # get the width and height of the image
    width, height = preoperative.shape[-2], preoperative.shape[-1]

    # setting the output image width and height
    output_height = int(numpy.ceil((height - occ_size) / occ_stride))
    output_width = int(numpy.ceil((width - occ_size) / occ_stride))

    # create a white image of sizes we defined
    heatmap = torch.zeros((output_height, output_width))

    # Initialize to do an "average" operation only:
    for name, p in model.named_parameters():
        if "bias" in name:
            p.data = torch.zeros_like(p)
        else:
            p.data = torch.ones_like(
                p) / (p.shape[1] * p.shape[-3] * p.shape[-2] * p.shape[-1])
        p.requires_grad = True

    model.init()

    baseOutput, _, _, _ = model(preoperative, intraoperative)

    print(
        "baseOutput:",
        torch.min(
            torch.abs(baseOutput)).item(),
        torch.max(baseOutput).item())

    with torch.no_grad():
        # iterate all the pixels in each column
        for h in range(0, height):
            print(str(h) + "/" + str(height))
            for w in range(0, width):

                h_start = h * occ_stride
                w_start = w * occ_stride
                h_end = min(height, h_start + occ_size)
                w_end = min(width, w_start + occ_size)

                if (w_end) >= width or (h_end) >= height:
                    continue

                preop = preoperative.clone().detach()

                # replacing all the pixel information in the image with
                # occ_pixel(grey) in the specified location
                preop[:, :, :, w_start:w_end, h_start:h_end] = occ_pixel

                # run inference on modified image
                output, _, _, _ = model(preop, intraoperative)

                val = output[0, :, 0, 0, 0].norm().item()
                heatmap[h, w] = val
                print(val)

                del preop, output

    print("heatmap", torch.min(heatmap), torch.max(heatmap))
    heatmap = heatmap - torch.min(heatmap)
    print("heatmap", torch.min(heatmap), torch.max(heatmap))
    heatmap = heatmap + 1e-32
    print("heatmap", torch.min(heatmap), torch.max(heatmap))
    heatmap_log = torch.log(heatmap)
    print("heatmap_log", torch.min(heatmap_log), torch.max(heatmap_log))
    heatmap = (heatmap - torch.min(heatmap)) / \
        (torch.max(heatmap) - torch.min(heatmap))
    print("heatmap", torch.min(heatmap), torch.max(heatmap))
    heatmap_log = (heatmap_log - torch.min(heatmap_log)) / \
        (torch.max(heatmap_log) - torch.min(heatmap_log))

    torchvision.utils.save_image(
        heatmap,
        "occlusionExperiment.png",
        normalize=True)
    torchvision.utils.save_image(
        heatmap_log,
        "occlusionExperiment_log.png",
        normalize=True)


def timingExperiment(model):

    st = time.time()
    numRuns = 250
    print("Timing test. Running {} samples...".format(numRuns))
    for i in range(numRuns):
        preoperative = torch.randn((1, 1, 64, 64, 64)).cuda()
        intraoperative = torch.randn((1, 1, 64, 64, 64)).cuda()
        preoperative.requires_grad = True
        intraoperative.requires_grad = True

        modelOutput, _, _, _ = model(preoperative, intraoperative)
        torch.cuda.synchronize()
        modelOutput.data.zero_()
    dt = time.time() - st
    print("\tTime:", dt)
    print(
        "\tTime per sample: {}, ({:2.2f} FPS)".format(
            dt / numRuns,
            numRuns / dt))
