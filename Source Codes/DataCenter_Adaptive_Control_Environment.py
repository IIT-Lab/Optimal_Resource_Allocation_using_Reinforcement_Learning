import random
import numpy as np
length_of_a_timestep = 300
charging_efficiency_of_the_battery = 0.7#----random
discharging_efficiency_of_the_battery = 0.8#--random
max_charging_rate = 0#----------------------per unit time
max_discharging_rate = 0#-------------------per unit time
max_charging_rate_per_timestep = length_of_a_timestep * max_charging_rate#---------random, in terms of energy
max_discharging_rate_per_timestep = length_of_a_timestep * max_discharging_rate#-------in terms of energy
BSOC_max = 5000#---------random
BSOC_min = 2500#---------0
coefficient_vector = np.array([charging_efficiency_of_the_battery, discharging_efficiency_of_the_battery, 0])
energy_cost_per_unit = 0.20
peak_power_cost_per_unit = 20
battery_degradation_cost_per_unit_energy = 0.05
constraint_loss_weight = 100000
energy_cost_per_unit = 0.20#arbitrary
peak_power_cost_per_unit = 20#arbitrary
battery_degradation_cost_per_unit_energy = 0.05#arbitrary
constraint_loss_weight = 100000#arbitrary

class DataCenter_Env:
    def __init__(self):
        #--------------State Variables
        self.DCI_demand_for_next_timestep = None#---------to be given by LSTM
        self.DCI_demand_for_this_slot = None#-------------to be given by LSTM
        self.DCI_demand_till_peak_end = None#-------------to be given by LSTM
        self.combined_demand_for_next_timestep = None#-----------------------------PROBLEM
        self.combined_demand_for_next_timestep = None#-----------------------------PROBLEM----Dt+It hola
        self.max_peak_demand_average = 0#--------------0 in the beginning
        self.timestep = 1#-----------------------------first timestep
        self.slot = 1#---------------------------------first slot
        self.current_BSOC = BSOC_max
        self.action_array = np.zeros((1,3))
        self.total_reward = 0
        self.peak_demand_buffer_for_a_slot = []
        self.max_avg_peak_demand_comparision_variable = 0

        #--------------Constants


        #---------------Other variables in terms of a timestep
        self.utility_energy_supply_to_DCI = 0
        self.utility_energy_supply_to_battery_for_next_timestep = 0
        self.actual_utility_energy_supply_to_battery = charging_efficiency_of_the_battery * self.utility_energy_supply_to_battery_for_next_timestep
        self.battery_energy_supply_to_DCI = 0
        self.actual_battery_energy_supply_to_DCI = discharging_efficiency_of_the_battery * self.battery_energy_supply_to_DCI

        self.DCI_demand_fulfilled_by_lstm = self.utility_energy_supply_to_DCI + self.battery_energy_supply_to_DCI
        self.utility_energy_supply_to_DCI_for_next_timestep = 0
        self.utility_energy_supply_to_battery_for_next_timestep = 0
        self.actual_utility_energy_supply_to_battery = charging_efficiency_of_the_battery * self.utility_energy_supply_to_battery_for_next_timestep
        self.battery_energy_supply_to_DCI_for_next_timestep = 0
        self.actual_battery_energy_supply_to_DCI = discharging_efficiency_of_the_battery * self.battery_energy_supply_to_DCI_for_next_timestep
        self.DCI_demand_fulfilled_by_lstm = self.utility_energy_supply_to_DCI_for_next_timestep + self.battery_energy_supply_to_DCI_for_next_timestep




        #self.states = [[1,1], [1,0], [0,1], [0,0]]
        #self.current_state = []

    def step(self, action_vector):#--------------------------action comes as a vector

        done = False

        #Take action and update BSOC
        self.current_BSOC = np.dot(action_vector, coefficient_vector)  # ------1)BSOC

        #update according to timeslot and timestep
        if (self.timestep == 3):
            #calculate new average peak demand max
            self.max_avg_peak_demand_comparision_variable = np.average(np.array(self.peak_demand_buffer_for_a_slot))


            #----------------Need to update Ymax before calculating self's reward for this ongoing slot
            if self.max_peak_demand_average < self.max_avg_peak_demand_comparision_variable:
                self.max_peak_demand_average = self.max_avg_peak_demand_comparision_variable#--------2)Ymax


            self.peak_demand_buffer_for_a_slot = [] #empty the buffer

            self.timestep = 1

            if self.slot == 24:

                self.slot = 1  # ----------------------------------------------3)Slot No

                #put the reward for the day in some new buffer here--------------------------VVI- IMP
                #update other variables accordingly

                done = True  # tells whether its time to reset the environment or not,
                # basically tells if the episode ended or not
                #we end the episode after each day, but continue the calculation of LSTM such that
                # we continue from 6pm the next day

            else:
                self.slot = self.slot + 1

        else:
            #store the timestep demand value
            self.peak_demand_buffer_for_a_slot.append(action_vector[2])# action vector's third element is Dt
            self.timestep = self.timestep + 1  # -------------------------------4)time step no



        #generate reward for the action taken since Ymax updated
        #generate reward for the action taken since the last slot/ Ymax updated
        reward = self.get_reward()
        self.total_reward = self.total_reward + reward

        #get new variables from the LSTM for the new state
        self.DCI_demand_for_next_timestep = 0 #--------------get from LSTM#------------5)
        self.DCI_demand_for_this_slot = 0 #------------------get from LSTM#------------6)
        self.DCI_demand_till_peak_end = 0 #------------------get from LSTM#------------7)

        #------------------------find some method to not use this when the peak ends



        return reward, done



    def reset(self):
        self.current_BSOC = BSOC_max
        #For the following code to be levant, you should call this reset right before 6pm the next day
        # get new variables from the LSTM for the new state
        self.DCI_demand_for_next_timestep = 0  # --------------get from LSTM#------------5)
        self.DCI_demand_for_this_slot = 0  # ------------------get from LSTM#------------6)
        self.DCI_demand_till_peak_end = 0  # ------------------get from LSTM#------------7)
        #need not update timestep, timeslot, Ymax since already taken care of

    def get_curent_state(self):
        return (self.DCI_demand_for_next_timestep,self.DCI_demand_for_this_slot,self.DCI_demand_till_peak_end,
                self.current_BSOC,self.max_peak_demand_average,self.timestep,self.slot)


    def get_reward(self):

        #basic reward model
        reward_original = energy_cost_per_unit * self.utility_energy_supply_to_battery_for_next_timestep + peak_power_cost_per_unit * self.max_peak_demand_average + battery_degradation_cost_per_unit_energy + self.battery_energy_supply_to_DCI
        reward_original = energy_cost_per_unit * self.utility_energy_supply_to_battery_for_next_timestep + peak_power_cost_per_unit * self.max_peak_demand_average + battery_degradation_cost_per_unit_energy + self.battery_energy_supply_to_DCI_for_next_timestep

        #variables to be used for added reward terms
        power_cap_constraint = self.combined_demand_for_next_timestep - (self.DCI_demand_for_next_timestep + self.utility_energy_supply_to_battery_for_next_timestep)

        DCI_demand_constraint = self.combined_demand_for_next_timestep - (self.DCI_demand_for_next_timestep + self.actual_battery_energy_supply_to_DCI)

        charging_capacity_constraint = (BSOC_max - self.current_BSOC) - self.actual_utility_energy_supply_to_battery

        discharging_capacity_constraint = (self.battery_energy_supply_to_DCI - self.current_BSOC)

        charging_rate_constraint = self.actual_utility_energy_supply_to_battery - max_charging_rate_per_timestep

        discharging_rate_constraint = self.battery_energy_supply_to_DCI - max_discharging_rate_per_timestep
        discharging_capacity_constraint = (self.battery_energy_supply_to_DCI_for_next_timestep - self.current_BSOC)

        charging_rate_constraint = self.actual_utility_energy_supply_to_battery - max_charging_rate_per_timestep

        discharging_rate_constraint = self.battery_energy_supply_to_DCI_for_next_timestep - max_discharging_rate_per_timestep

        #added reward for constraints
        reward_constraints = constraint_loss_weight * (power_cap_constraint + DCI_demand_constraint +charging_capacity_constraint+
                                                       discharging_capacity_constraint+charging_rate_constraint+
                                                       discharging_rate_constraint)
        #added reward for not meeting minimum feasible BSOC at the end of the peak period
        peak_end_reward = 0 #lets keep zero for now

        #also, since these rewards are not in the form of similar units, need to weight all three of these later

        return -(reward_original + reward_constraints + peak_end_reward)