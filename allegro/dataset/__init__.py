from torchvision import transforms
from transformers import T5Tokenizer

from allegro.dataset.allegro_datasets import Allegro_dataset
from allegro.dataset.transform import ToTensorVideo, TemporalRandomCrop, CenterCropResizeVideo

def getdataset(args):
    temporal_sample = TemporalRandomCrop(args.num_frames)
    norm_fun = transforms.Lambda(lambda x: 2. * x - 1.)
    if args.dataset == 't2v':
        transform = transforms.Compose([
            ToTensorVideo(),
            CenterCropResizeVideo((args.max_height, args.max_width)), 
            norm_fun
        ])
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir)
        return Allegro_dataset(args, transform=transform, temporal_sample=temporal_sample, tokenizer=tokenizer)
    raise NotImplementedError(args.dataset)