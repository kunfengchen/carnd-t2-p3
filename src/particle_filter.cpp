/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// DONE: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;

	std::default_random_engine gen;
	std::normal_distribution<double> N_x_init(x, std[0]);
	std::normal_distribution<double> N_y_init(y, std[1]);
	std::normal_distribution<double> N_theta_init(theta, std[2]);

	for (int i=0; i<num_particles; i++) {
		Particle p;
		p.id = i;
		p.weight = 1;
		p.x = N_x_init(gen);
		p.y = N_y_init(gen);
		p.theta = N_theta_init(gen);
        particles.push_back(p);
	}

	weights.resize(num_particles);
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// DONE: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine gen;
	for (Particle p: particles) {
		if (fabs(yaw_rate) > 0.001) {
			p.x = p.x + velocity/yaw_rate *
						(sin(p.theta+yaw_rate*delta_t) - sin(p.theta));
			p.y = p.y + velocity/yaw_rate *
						(cos(p.theta) - cos(p.theta+yaw_rate*delta_t));
		} else {  // yaw_rate is 0
			p.x = p.x + velocity*delta_t*cos(p.theta);
			p.y = p.y + velocity*delta_t*sin(p.theta);
		}
		p.theta = p.theta + yaw_rate*delta_t;

		std::normal_distribution<double> dist_x(p.x, std_pos[0]);
		std::normal_distribution<double> dist_y(p.y, std_pos[1]);
		std::normal_distribution<double> dist_theta(p.theta, std_pos[2]);

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// DONE: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (int i=0; i<observations.size(); i++) {
		double c_dist = 0.0; // the current distance
		double m_dist = 0.0; // track the minimum distance
		int m_dist_id = 0; // track the minimum distance id

		m_dist = dist(observations[i].x, observations[i].y,
					  predicted[0].x, predicted[0].y);
		for (int j=1; j<predicted.size(); j++) {
		    c_dist = dist(observations[i].x, observations[i].y,
			              predicted[j].x, predicted[j].y);
			if (m_dist > c_dist) {
				m_dist = c_dist;
				m_dist_id = j;
			}
		}

		observations[i].id = m_dist_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	// transformed to map coordinate
	std::vector<LandmarkObs> trans_observation(observations.size());
	// extract map landmarks
	std::vector<LandmarkObs> landmark_observation(map_landmarks.landmark_list.size());
	for (int i=0; i<landmark_observation.size(); i++) {
		landmark_observation[i].id = 0;
		landmark_observation[i].x = map_landmarks.landmark_list[i].x_f;
		landmark_observation[i].y = map_landmarks.landmark_list[i].y_f;
	}

	double new_weight = 1.0;
	double p_x, p_y, l_x, l_y, norm_x, norm_y, new_w, denom;
	for (int p=0; p<particles.size();p++) {
		// translate to map coordinate
        p_x = particles[p].x;
		p_y = particles[p].y;
		for (int i=0; i<observations.size(); i++) {
			trans_observation[i].x =
			    observations[i].x*cos(particles[p].theta) -
				observations[i].y*sin(particles[p].theta) + p_x;
			trans_observation[i].y =
			    observations[i].x*sin(particles[p].theta) +
				observations[i].y*cos(particles[p].theta) + p_y;
		}
		// Assosiate the nearest neighbors
		dataAssociation(landmark_observation, trans_observation);

		// Update the weights
        new_weight = 1.0;
		for (auto trans : trans_observation) {
			l_x = landmark_observation[trans.id].x;
			l_y = landmark_observation[trans.id].y;

			denom = 1.0/(2.0*M_PI*std_landmark[0]*std_landmark[1]);
            norm_x = ((trans.x-l_x)*(trans.x-l_x)) /
					(2*std_landmark[0]*std_landmark[0]);
			norm_y = ((trans.y-l_y)*(trans.y-l_y)) /
					(2*std_landmark[1]*std_landmark[1]);
            new_w = denom*exp(-1*(norm_x+norm_y));
			new_weight *= new_w;
		}
		particles[p].weight = new_weight;
		weights[p]= new_weight;
	}
}

void ParticleFilter::resample() {
	// DONE: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // std::random_device rd;
	// std::mt19937 gen(rd());
	// std::discrete_distribution<double> d

	std::vector<Particle> new_particles;
	double max_weights = 0;
	double beta = 0.0;
	int index = rand() % num_particles;

	// Get the maximum weight
	for (Particle p: particles) {
	    max_weights = max_weights >= p.weight? max_weights:p.weight;
    }

	for (int i=0; i<num_particles; i++) {
		beta += (rand() % 1)*(max_weights * 2.0);
	    while (beta > particles[index].weight) {
	     	beta -= particles[index].weight;
		    index = (index +1) % num_particles;
	    };
		Particle p;
		p.x = particles[index].x;
		p.y = particles[index].y;
		p.theta = particles[index].theta;
	    new_particles.push_back(p);
	}
	particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
