class ThresholdCalculator:
    def __init__(self, u1, u2, u3, u4, u11, u22, u33, u44 ,double):
        self.u2_1 = u1
        self.u2_2 = u2
        self.u2_3 = u3
        self.u2_4 = u4
        self.u1_1 = u11
        self.u1_2 = u22
        self.u1_3 = u33
        self.u1_4 = u44
        self.double = double
        self.calculate_deltas()

    def calculate_deltas(self):
        # Calculate deltas
        self.tail_threshold_delta1 = (self.u2_1 - self.u1_1) / self.double
        self.tail_threshold_delta2 = (self.u2_2 - self.u1_2) / self.double
        self.tail_threshold_delta3 = (self.u2_3 - self.u1_3) / self.double
        self.tail_threshold_delta4 = (self.u2_4 - self.u1_4) / self.double

    def print_results(self):
        print(f"self.tail_threshold_delta1: {self.tail_threshold_delta1}")
        print(f"self.tail_threshold_delta2: {self.tail_threshold_delta2}")
        print(f"self.tail_threshold_delta3: {self.tail_threshold_delta3}")
        print(f"self.tail_threshold_delta4: {self.tail_threshold_delta4}")
        print(f"double: {self.double}")

# Input values  myself 500-500算出来的
u1_1 = 0.5
u1_2 = 1.5
u1_3 = 2.5
u1_4 = 3.0

double = 5

u2_1 = 10.5
u2_2 = 14
u2_3 = 16
u2_4 = 17.5



# Create an instance of ThresholdCalculator and print results
calculator = ThresholdCalculator(u2_1, u2_2, u2_3,u2_4, u1_1, u1_2, u1_3,u1_4, double)
calculator.print_results()