import numpy as np
#define immutable constants/ expressions here
length_of_a_timestep = 300 #300 seconds/ 5 minutes
charging_efficiency_of_the_battery = 0.7#----random
discharging_efficiency_of_the_battery = 0.8#--random
max_charging_rate = 0#-------randomly placing zero for now, dont know for sure--------------- in terms of per unit time
max_discharging_rate = 0#------randomly placing zero for now-------------in terms of per unit time
max_charging_rate_per_timestep = length_of_a_timestep * max_charging_rate#---------random, in terms of energy for now
max_discharging_rate_per_timestep = length_of_a_timestep * max_discharging_rate#-------in terms of energy for now
BSOC_max = 5000#---------random
BSOC_min = 2500#---------0
coefficient_vector = np.array([charging_efficiency_of_the_battery, discharging_efficiency_of_the_battery, 0])
energy_cost_per_unit = 0.20
peak_power_cost_per_unit = 20
battery_degradation_cost_per_unit_energy = 0.05
large_coefficient_for_constraints = 100000

class DataCenter_Env:
    def __init__(self):
        #--------------State Variables
        self.DCI_demand_for_next_timestep = None#---------to be given by LSTM
        self.DCI_demand_for_this_slot = None#-------------to be given by LSTM
        self.DCI_demand_till_peak_end = None#-------------to be given by LSTM

        #removed self.combined demand of the battery and the DCI

        self.timestep = 1#-----------------------------first timestep
        self.slot = 1#---------------------------------first slot
        self.current_BSOC = BSOC_max
        self.action_array = np.zeros((1,3))

        self.utility_demand_buffer_for_timesteps_in_a_slot = []#getting the average of three values on this list gives the average of the slot
        self.max_avg_peak_demand_comparision_variable = 0#we put the average of the aboe list in this variable for comparison
        self.old_avg_peak_demand_in_a_slot = 0# we compare the above variable with this variable
        self.max_peak_demand_average = 0  # --------------0 in the beginning, we replace this value with the max of above two variables

        self.monetary_cost_without_peak_cost_in_a_day = []
        self.monetary_cost_with_peak_cost = 0

        self.day_counter_for_only_monetary_cost = 0

        self.reward_daywise = []#---------------------------------------------list of everyday reward
        self.cumulative_reward_in_a_day = []#-------------------------------->total accumulated reward till this day for the month, we just sum the elements of the above list




        # added penalty for not meeting minimum feasible BSOC at the end of the peak period
        self.peak_end_penalty_for_residual_energy_at_12am = 0  # lets keep zero for now

        #values to be given by RL algorithm, repalaced by action vector in this file
        self.Xt = 0#--------------------------->Summation of Ot and Dt
        self.It = 0#--------------------------->Energy to be supplied to the battery at next timestep
        self.Ot = 0#--------------------------->Energy to be discharged from the battery at next timestep

        #--------------Constants


        #---------------Other variables in terms of a timestep
        self.utility_energy_supply_to_DCI_for_next_timestep = 0#----------------------->Dt
        self.utility_energy_supply_to_battery_for_next_timestep = 0#------------------->It
        self.actual_utility_energy_supply_to_battery = charging_efficiency_of_the_battery * self.utility_energy_supply_to_battery_for_next_timestep
        self.battery_energy_supply_to_DCI_for_next_timestep = 0#----------------------->Ot
        self.actual_battery_energy_supply_to_DCI = discharging_efficiency_of_the_battery * self.battery_energy_supply_to_DCI_for_next_timestep

        self.DCI_demand_fulfilled_by_RL_agent = self.utility_energy_supply_to_DCI_for_next_timestep + self.actual_battery_energy_supply_to_DCI_for_next_timestep
        #The above value should match the LSTM's prediction for Xt and the difference should be made a penalty term in the reward model


    def step(self, action_vector):#--------------------------action comes as a vector from RL agent

        #done = False



        #Say LSTM exists and gives us the demand for timestep, slot and day = ? ? ? ---------------------
        self.DCI_demand_for_next_timestep = None  # ---------to be given by LSTM
        self.DCI_demand_for_this_slot = None  # -------------to be given by LSTM
        self.DCI_demand_till_peak_end = None  # -------------to be given by LSTM

        #------------------------find some method to not use this LSTM formulation when the peak ends until 6pm the next day




        #Say DDPG exists and gives us the action vector as
        self.action_array = action_vector


        #update the avg demand per slot buffer using It and Dt.
        self.utility_demand_buffer_for_timesteps_in_a_slot.append(action_vector[0] + action_vector[2])

        #Lets calculate the new BSCOC using the values of the action vector
        self.current_BSOC = np.dot(action_vector, coefficient_vector)  # ------1)BSOC#------------------------------>

        #update according to timeslot and timestep
        if (self.timestep == 3):



            self.max_avg_peak_demand_comparision_variable = np.average(np.array(self.utility_demand_buffer_for_timesteps_in_a_slot)) #assuming lstm le accurate dinxa address It and Dt


            #----------------Need to update Ymax before calculating self's reward for this ongoing slot
            if self.max_peak_demand_average < self.max_avg_peak_demand_comparision_variable:
                self.max_peak_demand_average = self.max_avg_peak_demand_comparision_variable#--------2)Ymax


            self.utility_demand_buffer_for_timesteps_in_a_slot = [] #empty the buffer

            self.timestep = 1

            self.cumulative_reward_in_a_day.append(self.get_reward())#maintain a list until the day ends and then append the sum of its elements to the list reward_daywise




            if self.slot == 24:



                #put the reward for the day in some new buffer here--------------------------VVI- IMP
                #update other variables accordingly
                #updataa BSOC

                reward = self.get_reward()
                self.cumulative_reward_in_a_day.append(reward)
                self.reward_daywise.append(np.sum(self.cumulative_reward_in_a_day))
                self.cumulative_reward_in_a_day=[]

                self.slot = 1  # ----------------------------------------------3)Slot No
                return reward



                #done = True  # tells whether its time to reset the environment or not,
                # basically tells if the episode ended or not
                #we end the episode after each day, but continue the calculation of LSTM such that
                # we continue from 6pm the next day

            else:

                reward = self.get_reward()
                self.cumulative_reward_in_a_day.append(reward)
                self.slot = self.slot + 1
                return reward



        else:
            #store the timestep demand value
            self.utility_demand_buffer_for_timesteps_in_a_slot.append(action_vector[0] + action_vector[2])

            reward = self.get_reward()
            self.cumulative_reward_in_a_day.append(reward)
            self.timestep = self.timestep + 1  # -------------------------------4)time step no
            return reward




    def reset(self):#-----------------------------------------------------------Reset some variables after peak period ends at 12am
        self.current_BSOC = BSOC_max
        #For the following code to be relevant, you should call this reset right before 6pm the next day
        # get new variables from the LSTM for the new state
        self.DCI_demand_for_next_timestep = 0  # --------------get from LSTM#------------5)
        self.DCI_demand_for_this_slot = 0  # ------------------get from LSTM#------------6)
        self.DCI_demand_till_peak_end = 0  # ------------------get from LSTM#------------7)

        self.day_counter_for_only_monetary_cost = 0
        #need not update timestep, timeslot, Ymax since already taken care of

    def get_current_state(self):
        return (self.DCI_demand_for_next_timestep,self.DCI_demand_for_this_slot,self.DCI_demand_till_peak_end,
                self.current_BSOC,self.max_peak_demand_average,self.timestep,self.slot)


    def get_reward(self):

        #basic reward model

        energy_consumption_cost = energy_cost_per_unit * (self.action_array[0]+ self.action_array[2])
        peak_demand_cost = peak_power_cost_per_unit * self.max_peak_demand_average
        battery_degradation_cost = battery_degradation_cost_per_unit_energy * self.action_array[1]

        reward_original = energy_consumption_cost + peak_demand_cost + battery_degradation_cost

        monetary_cost_without_peak_cost_per_timestep = energy_consumption_cost + battery_degradation_cost
        self.monetary_cost_without_peak_cost_in_a_day.append(monetary_cost_without_peak_cost_per_timestep)



        if(self.timestep == 3 and self.slot == 24):

            self.peak_end_penalty_for_residual_energy_at_12am = large_coefficient_for_constraints * abs(self.current_BSOC - self.action_array[1])# if anything is left more than the demand for the next time step, penalize

            self.day_counter_for_only_monetary_cost = self.day_counter_for_only_monetary_cost + 1

            print("The monetary cost at the end of day ", self.day_counter_for_only_monetary_cost, " is ", monetary_cost_without_peak_cost_per_timestep)

            if (self.day_counter_for_only_monetary_cost == 30):

                self.monetary_cost_with_peak_cost = np.sum(self.monetary_cost_without_peak_cost_in_a_day) + self.max_peak_demand_average * peak_demand_cost

                print("The monetary cost incurred at the end of 30 days is ", self.monetary_cost_with_peak_cost)

        #variables to be used for added reward terms
        DCI_demand_constraint = self.DCI_demand_for_next_timestep - (self.action_array[1] + self.action_array[2]) #------>LSTM value - The value suggested by RL

        charging_capacity_constraint = (BSOC_max - self.current_BSOC) - self.actual_utility_energy_supply_to_battery

        
        discharging_capacity_constraint = (self.battery_energy_supply_to_DCI_for_next_timestep - self.current_BSOC) #do not use the actual supply to the DCI variable here

        charging_rate_constraint = self.actual_utility_energy_supply_to_battery - max_charging_rate_per_timestep

        discharging_rate_constraint = self.battery_energy_supply_to_DCI_for_next_timestep - max_discharging_rate_per_timestep

        #added reward for constraints
        reward_constraints = large_coefficient_for_constraints * (DCI_demand_constraint + charging_capacity_constraint +#-------------->all constraints should be satisfied at all
                                                                  discharging_capacity_constraint + charging_rate_constraint +#---->times, so equal and large coefficient given
                                                                  discharging_rate_constraint)#-------------------------------------> for all of them.




        #also, since these rewards are not in the form of similar units, need to weight all three of these later#----------------------> VVI



        reward_to_return = -(reward_original + reward_constraints + self.peak_end_penalty_for_residual_energy_at_12am)
        self.peak_end_penalty_for_residual_energy_at_12am = 0

        return reward_to_return