MODEL_CONFIG = {
    'unet':{
        'resnet50_dropout':{
            'in_channels':1,
            'encoder_name':'resnet50_dropout',
            'encoder_depth':5,
            'encoder_channels':[64,256,512,1024,2048],  #[2,4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[16,8,4,2]
            'upsampling':2,
            'classes':2,
            'aux_classifier': False,
            'aux_losspredictor': False
        },
        'resnet50_naive':{
            'in_channels':1,
            'encoder_name':'resnet50_naive',
            'encoder_depth':5,
            'encoder_channels':[64,256,512,1024,2048],  #[2,4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[16,8,4,2]
            'upsampling':2,
            'classes':2,
            'aux_classifier': False,
            'aux_losspredictor': False
        }
    }
}