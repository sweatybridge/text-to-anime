# text-to-anime

Convert text and audio to facial expressions

## Data preparation

1. Download [LRS3 dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html) to `lrs3_v0.4` directory
2. Download TED talks from YouTube to `video/{id}.mp4` where id is the query parameter `v`
   - https://www.youtube.com/watch?v=0C5UQbWzwg8
   - https://www.youtube.com/watch?v=0FQXicAGy5U
   - https://www.youtube.com/watch?v=0FkuRwU8HFc
   - https://www.youtube.com/watch?v=0GL5r3HVAZ0
   - https://www.youtube.com/watch?v=0LxPAY9yis8
   - https://www.youtube.com/watch?v=0akiEFwtkyA
   - https://www.youtube.com/watch?v=0bop3D7SdDM
   - https://www.youtube.com/watch?v=0d6iSvF1UmA
   - https://www.youtube.com/watch?v=0hzSUUdTDUA
   - https://www.youtube.com/watch?v=0iTehgSOZ8A
3. Extract annotated frames from download videos to `noisy` directory

```bash
python preprocess.py
```

4. Detect facial landmarks using OpenFace 2.0

```bash
docker-compose up
```

5. Copy high confidence detections to `clean` directory

```bash
python postprocess.py
```

## Training

The `clean` directory contains sample data that have been preprocessed. You may use it to reproduce our model.

```bash
python train.py
```

## Inference

1. Call the model with text input and save the numpy array returned (see `text_to_anime.ipynb` notebook for an example)
2. Combine the output displacement trajectories with a reference frame to generate facial landmarks.

```bash
python face.py
```
