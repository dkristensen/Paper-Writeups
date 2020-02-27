# Writeup of Paper
### Title: A Connection Between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models

#### Authors: Chelsea Fin∗ , Paul Christiano , Pieter Abbeel, Sergey Levine
###### Organization: University of California, Berkeley

---
# Takeaways from abstract
* Learning cost functions is a new idea in GANs, but not in RL (especially IL/IRL)
* There is an equivalence in cost learning in certain IRL methods to that of GANs
* Direct equivalence between a sample based approach for MaxEnt IRL and GANs with evaluable density functions which input into the descrimantor
## Introduction
* Gradient updates for the cost and policy in a MaxEntIRL model can be viewed as updates to the discriminator and generators for these specific subset of GANs
	* Examples of GANs which fall in this category are autoregressive models
* The implication of this is that training using a GAN approach is preferable even when density values are tractable
	* Example given is a network with insufficient capacity to represent a multinomial distribution; a maximized likelihood would cover the modes, but place most of the mass in low density regions of the data distribution, whereas an adversarial approach would minimize this mislead apportionment of mass with the trade off of lower diversity
* "GAN training can significantly improve the quality of samples even when the generator density can be exactly evaluated"
* Since MaxEntIRL models are special cases of energy based models, certain GANs can be used to train EBMs

## GANs and IRL
### Which kinds of discriminators work for this equivalence
* Typical discriminator output is  <br/> $D^*(\tau) = \frac{p(\tau)}{p(\tau)+q(\tau)}$
with $p(\tau)$ the actual distribution of the data and $q(\tau)$ the generator's density
* If the generator density is known, it is possible to modify the output of the discriminator to output $p_\theta(\tau)$ instead, and fill in the known $q(\tau)$, such that our discriminator takes the new form <br/> $D_\theta (\tau) = \frac{\tilde{p}_\theta(\tau)}{\tilde{p}_\theta(\tau) + q(\tau)}$.
* Since in a Boltzmann distribution, the energy of a trajectory is given by $p_\theta(\tau) = \frac{1}{Z}\text{exp}(-c_\theta(\tau))$, we can represent this new discriminator output to be <br/> $D_\theta (\tau) = \frac{\frac{1}{Z}\text{exp}(-c_\theta(\tau))}{\frac{1}{Z}\text{exp}(-c_\theta(\tau)) + q(\tau)}$.
* This resembles the form of a binary classification model with a sigmoid output, only with a subtraction of $\log(q(\tau))$, meaning the optimal discriminator is independent of the generator - something which leads to potentially increasing stability of training
 ---
### Proving equivalence
The discriminator's loss is given by <br/> $L_\text{discriminator}(D_\theta) = \mathbb{E}_{\tau 		\sim p}\left[-\log\frac{\frac{1}{Z}\text{exp}(-c_\theta(\tau))}{\frac{1}{Z}\text{exp}(-c_\theta(\tau)) + q(\tau)}\right] + \mathbb{E}_{\tau \sim q}\left[-\log\frac{q(\tau)}{\frac{1}{Z}\text{exp}(-c_\theta(\tau)) + q(\tau)}\right]$.
The MaxEntIRL log-likelihood objective is <br/> $L_\text{cost}(D_\theta) = \mathbb{E}_{\tau 		\sim p}\left[c_\theta(\tau)\right] + \log\left( \mathbb{E}_{\tau \sim \mu}\left[\frac{\text{exp}(-c_\theta(\tau)))}{\frac{1}{2Z}\text{exp}(-c_\theta(\tau)) + \frac{1}{2}q(\tau)}\right]\right)$, with $\tilde{\mu}(\tau) = \frac{1}{2Z}\text{exp}(-c_\theta(\tau)) + \frac{1}{2}q(\tau)$

* ## Equivalence of Z
	Expanding the loss for the discriminator, we get <br/> $L_\text{discriminator}(D_\theta) =\log Z + \mathbb{E}_{\tau\sim p}[c_\theta(\tau)] - \mathbb{E}_{\tau\sim q}[\log q(\tau)] + 2\mathbb{E}_{\tau\sim\mu}[\log \tilde{\mu}(\tau)]$.
	Since the first and last terms are the only components with Z, at the minimizer for Z, $\partial_Z L_\text{discriminator}(D_\theta) = 0$, and $Z=\mathbb{E}_{\tau\sim\mu}[\frac{\text{exp}(-c_\theta(\tau))}{\tilde{\mu}(\tau)}]$, which is the second value in the loss for the MaxEntIRL model (the estimate of the importance sampling estimate of Z)

* ## Equivalence for $c_\theta$
	From the expanded disciminator loss in the section above, only the second and fourth terms use $\theta$. Thus,
	$\partial_\theta L_\text{discriminator}(D_\theta) = \mathbb{E}_{\tau\sim p}[\partial_\theta c_\theta(\tau)] - \mathbb{E}_{\tau\sim\mu}\left[\frac{\frac{1}{Z}\text{exp}(-c_\theta(\tau))\partial_\theta c_\theta(\tau)}{\tilde{\mu}(\tau)}\right]$.
	Differentiating the MaxEntIRL objective, we get
	$\partial_\theta L_\text{cost}(\theta) = \mathbb{E}_{\tau\sim p }[\partial_\theta c_\theta(\tau)] + \partial_\theta \log\left(\mathbb{E}_{\tau\sim\mu}\left[\frac{\text{exp}(-c_\theta(\tau))}{\tilde{\mu}(\tau)}\right]\right)$, which simplifies to $\partial_\theta L_\text{discriminator}(D_\theta)$.


As for the generator, it maximizes the sampler loss from the MaxEntIRL loss (see paper section 3.3)

---
## Discussion Section Points
* Equivalence between generative adversarial modeling and max entropy irl.
* Need the special form of GAN with tractable generator density
