

## Generate the hidden feature - helix radii as input 

* I find the most tricky part doing clustering in this competition is simulating the range of helix radii. We've tried different distributions, such as linear distribution, Gaussian distribution, but the most effective way is to generate real helix radii from the train data. I stole @Heng's code from [this post](https://www.kaggle.com/c/trackml-particle-identification/discussion/57643).
 
* You can find [my notebook](https://github.com/nicolefinnie/kaggle-trackml/blob/master/src/notebooks/generate_radii_samples.ipynb) which generates helix radii from the train data or you can download pre-generated radii named after their event id [here](https://github.com/nicolefinnie/kaggle-trackml/tree/master/input/r0_list)

## Helix unrolling function

### z-axis centered tracks, tracks crossing (x,y) = (0,0)
* The closet approach `D0 = 0`, the tracks start from close to `(0,0,z)`.

![helix unrolling](https://github.com/nicolefinnie/kaggle-trackml/blob/master/images/helix_unrolling.png)
* Accurate version - reach `score=0.5` within 1 minute with 40 radius samples. To get a higher accuracy, you need to run more radius samples and it can take much longer.

```
# The track can go in either direction, and theta0 should be constant for all hits 
# on the same track in a perfect helix form.

  dfh['cos_theta'] = dfh.r/2/r0

  if ii < r0_list.shape[0]/2:
      dfh['theta0'] = dfh['phi'] - np.arccos(dfh['cos_theta'])
  else:
      dfh['theta0'] = dfh['phi'] + np.arccos(dfh['cos_theta'])

```

* Self-made version - reach `score=0.5` in 2 minutes but it rarely go above 0.5 since the unrolling function is not accurate. The reason why we use this self-made version is that it can find different tracks for later merging, which is good.

```
# This tries to find possible theta0 using an approximation function
# ii from -120 to 120

   STEPRR = 0.03
   rr = dfh.r/1000
   dfh['theta0'] = dfh['phi'] + (rr + STEPRR*rr**2)*ii/180*np.pi + (0.00001*ii)*dfh.z*np.sign(dfh.z)/180*np.pi


```

#### Main features that are constant in a perfect helix form
* `sin(theta0)`
* `cos(theta0)`
* `(z-z0)/arc`

* The problem is `z/arc` is still uneven in the magnetic field, so I've been trying to improve this problem by using following features in different models. Other Kagglers definitely have a more accurate equation. 
* `log(1 + abs((z-z0)/arc))*sign(z)`
* `(z-z0)/arc*sqrt(sin(arctan2(r,z-z0)))` I use this square root sine function as a correction term for the azimuthal angle on the x-y plane projection

#### Main features that are often constant
* `(z-z0)/r` where `r` is the Euclidean distance from the hit to the origin.
* `log(1 + abs((z-z0)/r))*sign(z)` is an approach to get z values closer to the origin to improve the problem with uneven `z/r` values

#### Side features
* Those are often not constant but we can find different tracks using them with small weights when we cluster
* `x/d` where `d` is the Eucliean distance from the hit to the origin in 3D `(x**2+y**2+z**3)**0.5`
* `y/d`
* `arctan2(z-r0,r)`
* `px, py` in my code: `-r*cos(theta0)*cos(phi)-r*sin(theta0)*sin(phi)` and `-r*cos(theta0)*sin(phi)+r*sin(theta0)*cos(phi)`. I happened to find this feature that can find the seeds of non-z-axis centered tracks and we can extend the tracks using the found seeds. 

### Non-z-axis centered tracks
* Skip this part since we didn't have time to implement it. Add the closest approach `D0` to your equations, you can find full equations with `D0` and `D1` from [Full helix equations for ATLAS](http://www.hep.ucl.ac.uk/atlas/atlantis/files/helix_equations_1.pdf)


## LSTM approach for track fitting
* My [notebook](https://github.com/nicolefinnie/kaggle-trackml/blob/master/src/notebooks/train_LSTM.ipynb) including visualization
* We take first five hits as a seeded track and predict next 5 hits. I believe this has its potential as a track validation tool if right features were trained. I shared the detail in [this post](https://www.kaggle.com/c/trackml-particle-identification/discussion/60455#352645)

## pointNet approach 
* PointNet is a lightweight CNN that can be used for pixel level classification(segementation) or classification. I put experimental code [here](https://github.com/nicolefinnie/kaggle-trackml/blob/master/src/train_pointnet.py), the challenge is to generate the right train data.


## Background knowledge 

* [Very good slides for beginners](http://ific.uv.es/~nebot/IDPASC/Material/Tracking-Vertexing/Tracking-Vertexing-Slides.pdf)

* [Lecture of particles tracking](http://www.physics.iitm.ac.in/~sercehep2013/track2_Gagan_Mohanty.pdf)


* [Full helix equations for ATLAS](http://www.hep.ucl.ac.uk/atlas/atlantis/files/helix_equations_1.pdf) - All equations you need!


* [Diplom thesis](http://physik.uibk.ac.at/hephy/theses/dipl_as.pdf) of Andreas Salzburger (Wow, he started in this field as a CERN student already in 2001 :stuck_out_tongue_closed_eyes: )

* [Doctor thesis](http://physik.uibk.ac.at/hephy/theses/diss_as.pdf) of Andreas Salzburger

* [CERN tracking software Acts](https://gitlab.cern.ch/acts/acts-core) - Sadly, we didn't have time to explore it :) 


