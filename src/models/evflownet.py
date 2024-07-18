import torch
from torch import nn
from src.models.base import *
from typing import Dict, Any

_BASE_CHANNELS = 64

class EVFlowNet(nn.Module):
    def __init__(self, args):
        super(EVFlowNet,self).__init__()
        self._args = args

        self.encoder1 = general_conv2d(in_channels = 4, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder2 = general_conv2d(in_channels = _BASE_CHANNELS, out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder3 = general_conv2d(in_channels = 2*_BASE_CHANNELS, out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder4 = general_conv2d(in_channels = 4*_BASE_CHANNELS, out_channels=8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.resnet_block = nn.Sequential(*[build_resnet_block(8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm) for i in range(2)])

        self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16*_BASE_CHANNELS,
                        out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8*_BASE_CHANNELS+2,
                        out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4*_BASE_CHANNELS+2,
                        out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2*_BASE_CHANNELS+2,
                        out_channels=int(_BASE_CHANNELS/2), do_batch_norm=not self._args.no_batch_norm)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # encoder
        skip_connections = {}
        inputs = self.encoder1(inputs)
        skip_connections['skip0'] = inputs.clone()
        inputs = self.encoder2(inputs)
        skip_connections['skip1'] = inputs.clone()
        inputs = self.encoder3(inputs)
        skip_connections['skip2'] = inputs.clone()
        inputs = self.encoder4(inputs)
        skip_connections['skip3'] = inputs.clone()

        # transition
        inputs = self.resnet_block(inputs)

        # decoder
        flow_dict = {}
        inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
        inputs, flow = self.decoder1(inputs)
        flow_dict['flow0'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
        inputs, flow = self.decoder2(inputs)
        flow_dict['flow1'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
        inputs, flow = self.decoder3(inputs)
        flow_dict['flow2'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
        inputs, flow = self.decoder4(inputs)
        flow_dict['flow3'] = flow.clone()

        return flow

class EVFlowNetTransformer(nn.Module):
    def __init__(self, args):
        super(EVFlowNetTransformer, self).__init__()
        self._args = args

        self.embedding = nn.Linear(4, 64)  # 入力チャネル数を埋め込み次元に変換

        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        decoder_layer = nn.TransformerDecoderLayer(d_model=64, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.flow_predictor = nn.Linear(64, 2)  # 埋め込み次元をフローの次元に変換

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # 入力を埋め込み
        B, C, H, W = inputs.shape  # バッチサイズ、チャネル数、高さ、幅を取得
        inputs = inputs.view(B, C, -1).permute(0, 2, 1)  # (B, C, H*W) -> (B, H*W, C)
        inputs = self.embedding(inputs)  # (B, H*W, 4) -> (B, H*W, 64)

        # Transformerエンコーダ
        memory = self.transformer_encoder(inputs)

        # Transformerデコーダ
        outputs = self.transformer_decoder(inputs, memory)

        # フローの予測
        flow = self.flow_predictor(outputs)  # (B, H*W, 64) -> (B, H*W, 2)
        flow = flow.permute(0, 2, 1).view(B, 2, H, W)  # (B, H*W, 2) -> (B, 2, H, W)

        return {'flow': flow}

# 既存のコードのエンコーダとデコーダの部分をTransformerに置き換えました。
# 埋め込み層を追加し、Transformerエンコーダとデコーダを使用してフローを予測します。
        

# if __name__ == "__main__":
#     from config import configs
#     import time
#     from data_loader import EventData
#     '''
#     args = configs()
#     model = EVFlowNet(args).cuda()
#     input_ = torch.rand(8,4,256,256).cuda()
#     a = time.time()
#     output = model(input_)
#     b = time.time()
#     print(b-a)
#     print(output['flow0'].shape, output['flow1'].shape, output['flow2'].shape, output['flow3'].shape)
#     #print(model.state_dict().keys())
#     #print(model)
#     '''
#     import numpy as np
#     args = configs()
#     model = EVFlowNet(args).cuda()
#     EventDataset = EventData(args.data_path, 'train')
#     EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=args.batch_size, shuffle=True)
#     #model = nn.DataParallel(model)
#     #model.load_state_dict(torch.load(args.load_path+'/model18'))
#     for input_, _, _, _ in EventDataLoader:
#         input_ = input_.cuda()
#         a = time.time()
#         (model(input_))
#         b = time.time()
#         print(b-a)