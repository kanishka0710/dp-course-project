Channel Estimation with DP

1. Initialize variables
  - Initial Reflector positions
    - Channel Measurement
2. DP takes in estimated reflector positions and tries to predict the next channel measurement + action (confidence)
  - if confidence is low, we transition system back to probing
    - else keep serving and keep adding reward
      - beams are created using the channel estimate
3. Repeat

Serving/Probing with DP

We have initial value of the channel, and no knowledge of moving reflectors
The channel evolves as the reflectors move
We have the past history of interference powers from serving
We need to decide when it is worth probing