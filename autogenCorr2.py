# Import necessary modules
from langchain.chains import ConversationChain, LLMChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain_huggingface.llms import HuggingFacePipeline
import requests
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re

# Login with access key
access_key = "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"
login(token=access_key)

# Initialize LLM and memory
mod = "llama-3.2-11b-vision-preview"  # "llama-3.2-11b-text-preview"
llm = ChatGroq(temperature=0, groq_api_key="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", model_name=mod,
               max_tokens=8000)
memory = ConversationSummaryMemory(llm=llm)

# Define prompt templates for different use cases
prompts = {
    "verilog_counter": PromptTemplate(
        input_variables=["user_input"],
        template="""
          You are a Verilog Programmer. Retrieve required data correctly from <user_input>.
          First classify counter type: normal counter, ring counter, down counter and counter with enable.
          Read given <user_input> data care fully. Check if <input_name>, <output_name> or <module_name> is given.
          If <module_name> is not given, generate a name using <counter> and <bit_size>.
          Always Respond with the names provided by user.
          Read numbers carefully. Do not miss any number in the values.
          Choose correctly type from the following:
          Type 1: normal: implement a 2-bit counter:
                                    module counter_2bit(
                                            input CLK,
                                            input reset,
                                            output reg [1:0] count
                                        );
                                        always @(posedge CLK or negedge reset)
                                        begin
                                            if (!reset)
                                                count <= 2'b00;
                                            else
                                                count <= count + 1;
                                        end
                                        endmodule
          Type 2: with enable: implement a 4-bit counter with output bit_cnt and enable
                                    module counter_4bit(
                                            input CLK,
                                            input reset, input. enable, output reg [1:0] bit_cnt
                                        );
                                        always @(posedge CLK or negedge reset)
                                        begin
                                            if (!reset)
                                                bit_cnt <= 4'b0000;
                                            else
                                                bit_cnt <= bit_cnt + 1;
                                        end
                                        endmodule


          Type 3: ring counter: Write a 8-bit ring counter
                                    module ring_counter_4bit(
                                            input CLK,
                                            input reset,
                                            output reg [7:0] count
                                        );

                                        always @(posedge CLK or negedge reset)
                                        begin
                                            if (!reset)
                                                count <= 8'b0000;
                                            else
                                                count <= [count[6:0], count[7]];
                                        end

                                        endmodule

                  - Always usr curly brackets instead of square brackets to enclose count[6:0], count[7]. Replace the outter square brackets with curly brackets.

          Type 4: down counter:  write a 3-bit down counter
                                    module down_counter_3bit(
                                            input CLK,
                                            input reset,
                                            output reg [2:0] count
                                        );

                                        always @(posedge CLK or negedge reset)
                                        begin
                                            if (!reset)
                                                count <= 3'b0;
                                            else
                                                count <= count - 1;
                                        end
                                        endmodule

            - Following are some example:

              ###Instruction: 4 bit up-down counter with enable up_dn_en
              ###Respons: module up_down_counter_4bit(
                            input CLK,
                            input reset,
                            input enable,
                            output reg [3:0] count
                          );

                          always @(posedge CLK or negedge reset)
                          begin
                            if (!reset)
                            count <= 4'b0000;
                            else if (up_dn_en)
                            count <= count + 1;
                            else
                            count <= count - 1;
                          end

                          endmodule

              ###Instruction: write 4-bit ring counter
              ###Respons: module ring_counter_4bit(
                            input CLK,
                            input reset,
                            output reg [3:0] count
                          );

                          always @(posedge CLK or negedge reset)
                          begin
                            if (!reset)
                            count <= 4'b0000;
                            else
                            count <= [count[2:0], count[3]];
                          end

                          endmodule

              ###Instruction: write 8-bit ring counter with output ringc and async reset rstn
              ###Respons: module ring_counter_8bit(
                            input CLK,
                            input reset,
                            output reg [7:0] ringc
                          );

                          always @(posedge CLK or negedge rstn)
                          begin
                            if (!rstn)
                            ringc <= 8'b00000000;
                            else
                            ringc <= [ringc[6:0], ringc[7]];
                          end

                          endmodule


              ###Instruction: write 8-bit ring counter with output beta and sync reset alpha
              ###Responsez: module ring_counter_8bit(
                            input CLK,
                            input reset,
                            output reg [7:0] beta
                          );

                          always @(posedge CLK)
                          begin
                            if (!alpha)
                            beta <= 8'b00000000;
                            else
                            beta <= [beta[6:0], beta[7]];
                          end

                          endmodule

              ###Instruction: write 4 to 1 mux with output data_o and input data_i[3:$]
              ###Response:


          - User: {user_input}
          - Note: 1. Always take care of input name, output name, and module names.
                  2. Use same style and syntax as the examples. Calculations should be mathematically correct. Recheck mathematical calculations twice.
                  3. Verilog code should be in same sytax and style as above examples.
                  4. For Type 3: ring counter, make sure you name the ouput at correct place and use curly brackets.
                  4. Perform all calculation steps carefully.
                  5. Accuracy and precision are your priority.
                  6. Provide response only with single verilog code and nothing else.
          """
    ),
    "flip_flop": PromptTemplate(
        input_variables=["user_input"],
        template="""
          You are a Verilog Programmer. Retrieve required data correctly from <user_input>.
          - Always read given <user_input> data care fully. Check and identify if <input_name>, <output_name>, <reset_name>, <enable_name>, <clock_name> or <module_name> is/are given.
          - If <module_name> is not given, generate a name using <flip_type> and <bit_size>.
          - Always classify all the names mentioned in the user_input. Names are always initiated after their corresponding pin type.
          - Always look for potential names. Names can be anythin from words, names, number to mixture of both.
          - For names, always choose names using following rules:
              - Golden Rule: pin names are always initiated in the <user_input> followed by their corresponding pin type.
              - If <input_name> is given and <output_name> is not given, simply add "_ff" as suffix to <input_name> to create <output_name>. Do not change the given <input_name>. For example if input name is "data" the output name should be "data_ff".
              - If <output_name> is given and <input_name> is not given, remove any prefix or suffix like "out", "ff", "output", "outter", "open", "outwards" from <output_name> to create <input_name>. For example if <output_name> is "data_ff" the <input_name> should be "data".
              - If both <input_name> and <output_name> is given, use them as it is. Do not add "_ff" in this case.
          - Choose correct type from the following:
            Type 1: D flip-flop:
                                module d_flip_flop(
                                    input D,
                                    input CLK,
                                    output reg Q
                                );
                                always @(posedge CLK)
                                    Q <= D;
                                endmodule
            Type 2: with synchronous reset: following example is with 8-bit
                                          module reg_8bit_sync_reset(
                                                input [7:0] D,
                                                input CLK,
                                                input reset,
                                                output reg [7:0] Q
                                            );
                                            always @(posedge CLK)
                                            begin
                                                if (reset)
                                                    Q <= 8'b00000000;
                                                else
                                                    Q <= D;
                                            end
                                            endmodule


            Type 3: with asynchronous reset: following example is with 8-bit
                                          module reg_8bit_sync_reset(
                                                input [7:0] D,
                                                input CLK,
                                                input reset,
                                                output reg [7:0] Q
                                            );
                                            always @(posedge CLK or negedge resetn)
                                            begin
                                                if (!resetn)
                                                    Q <= 8'b00000000;
                                                else
                                                    Q <= D;
                                            end
                                            endmodule

            Type 4: with enable: For this type, Make sure to always generate output register value in the following format:  "<output_name><= 1'b0;".  following example is with 4-bit.
                              module reg_4bit_enable(
                                    input [3:0] D,
                                    input CLK,
                                    input enable,    input resetn,
                                    output reg [3:0] Q
                                );
                                always @(posedge CLK or negedge reset)
                                begin    if (!resetn)
                                        Q<= 1'b0;
                                  else
                                        Q <= D;
                                end
                                endmodule

        - Following are some example:
            ###Instruction: implement 4-bit flop with output datao and input data_in and async reset rst and enable alpha
            ###Respons: module reg_4bit_async_flop(
                              input [3:0] data_in,
                              input CLK,
                              input rst,
                              input alpha,
                              output reg [3:0] datao
                          );
                          always @(posedge CLK or negedge rst)
                          begin
                              if (!rst)
                                  datao <= 4'b0000;
                              else
                                  datao <= data_in;
                          end
                          endmodule

            ###Instruction: implement 12-bit flop with output vikram and input mehta and async reset rrr and enable poke
            ###Respons: module reg_12bit_async_flop(
                              input [11:0] mehta,
                              input CLK,
                              input rrr,
                              input poke,
                              output reg [11:0] vikram
                          );
                          always @(posedge CLK or negedge rrr)
                          begin
                              if (!rrr)
                                  vikram <= 12'b000000000000;
                              else
                                  vikram <= mehta;
                          end
                          endmodule

            ###Instruction: write 8-bit flop with synchronous reset, with input mamba and output apple
            ###Respons: module reg_8bit_sync_flop(
                                input [7:0] mamba,
                                input CLK,
                                input reset,
                                output reg [7:0] apple
                            );
                            always @(posedge CLK)
                            begin
                                if (reset)
                                    apple <= 8'b00000000;
                                else
                                    apple <= mamba;
                            end
                            endmodule


            ###Instruction: write a 4-bit flop with enable rwi, input name cell, output as gemini and reset as high
            ###Respons: module reg_4bit_enable(
                              input [3:0] cell,
                              input CLK,
                              input rwi,
                              input high,
                              output reg [3:0] gemini
                          );
                          always @(posedge CLK or negedge reset)
                          begin    if (!high)
                                  gemini<= 1'b0;
                            else
                                  gemini <= cell;
                          end
                          endmodule"

        - User: {user_input}
        - Analyse all steps and example before choose names and generating specific type of flip-flop.
        - Note: 1. Always take care of input name, output name, clock name, reset name, enable name and module names. Make sure if mentioned in user_input, should be correct.
                2. Use same style and syntax as the examples. Calculations should be mathematically correct. Recheck mathematical calculations twice.
                3. Verilog code should be in same sytax and style as above examples.
                4. Perform all steps carefully.
                5. Accuracy and precision are your priority.
                6. Provide response only with single verilog code and nothing else.
        """
    ),
    "only_mux": PromptTemplate(
        input_variables=["user_input"],
        template="""
        You are a Verilog code Provider. Retrieve required data correctly from user input.
        Read numbers carefully (specially number 5). Do not miss any number in the values.
        Get inputs <mux_type>, <input_width> and input/output names from user input and perform the following calculations:
            1. Retrieve <mux_type> from user input, correctly.
                - Examples: - for mux 3 to 1, 3:1, 3to1; <mux_type>=3,
                            - for mux 9 to 1; <mux_type>=9,
                            - for mux 35 to 1, 35:1, 35to1; <mux_type>=35,
                            - for mux 263 to 1, 263:1, 263to1; <mux_type>=263,
                            - for mux 1001 to 1; <mux_type>=1001 and so on.
            2. Calculate the <bit_size> based on the given mux type and input width, using this formula:
                - bit_size = <mux_type>-1
                - Put the calculated <bit_size> value in the code input declaration, correctly.

            3. Calculate range values for perticular pin use following formula:
                - pin__n_range = [<n>]
                - Always take care of number of pins <n>.

            4. Calculate select_pin count:
                - select_pin=log2(<mux_type>)       # log_base_2
                - Perform log base 2 and choose roof value.
                - Example:  1. If input is 9-to-1 mux, select_pin=log2(9)=4
                            2. If input is 3-to-1 mux, select_pin=log2(3)=2
                            3. If input is 8-to-1 mux, select_pin=log2(8)=3
                - Put the calculated <select_pin> value in the code input declaration, correctly.

            5. Put all calculated values in the code.
                - Note: Do not use keyword in code, use actual calculated values.

        - Samples codes:
          {{###Input: create/implement/write 8-to-1 mux with input <input_name>
              Calculation:  Here <mux_type> is 8
                            bit_size=8-1=7
                            pin_0_range=0
                            pin_1_range=1
                            pin_2_range=2
                            pin_3_range=3
                            pin_4_range=4
                            pin_5_range=5
                            pin_6_range=6
                            pin_7_range=7

                            <select_pin>=log2(8)=3
            ###Output:  module mux_8to1(
                          input [<bit_size>:0] <input_name>,
                          input [<select_pin>-1:0] select,
                          output reg output
                        );
                        always @(*)
                          case(select)
                            3'd0: output = <input_name>[<pin_0_range>];
                            3'd1: output = <input_name>[<pin_1_range>];
                            3'd2: output = <input_name>[<pin_2_range>];
                            3'd3: output = <input_name>[<pin_3_range>];
                            3'd4: output = <input_name>[<pin_4_range>];
                            3'd5: output = <input_name>[<pin_5_range>];
                            3'd6: output = <input_name>[<pin_6_range>];
                            3'd7: output = <input_name>[<pin_7_range>];
                            default: output = 'b0;
                          endcase
                        endmodule

            ###Instruction: create/implement/write 9-to-1 mux with input mm
              Calculation:  Here mux_type is 9
                            bit_size=9-1=8
                            pin_0_range=0
                            pin_1_range=1
                            pin_2_range=2
                            pin_3_range=3
                            pin_4_range=4
                            pin_5_range=5
                            pin_6_range=6
                            pin_7_range=7
                            pin_8_range=8

                            select_pin=log2(9)=3
            ###Response:  module mux_9to1(
                          input [8:0] mm,
                          input [2:0] select,
                          output reg output_data
                        );
                        always @(*)
                          case(select)
                            3'd0: output_data = mm[0];
                            3'd1: output_data = mm[1];
                            3'd2: output_data = mm[2];
                            3'd3: output_data = mm[3];
                            3'd4: output_data = mm[4];
                            3'd5: output_data = mm[5];
                            3'd6: output_data = mm[6];
                            3'd7: output_data = mm[7];
                            3'd8: output_data = mm[8];
                            default: output_data = 'b0;
                          endcase
                        endmodule

            ###Instruction: write 4:1 mux with output <output_name> and input <input_name>
              Calculations:  Here mux_type is 4
                            bit_size=4-1=3
                            pin_0_range=0
                            pin_1_range=1
                            pin_2_range=2
                            pin_3_range=3
                            select_pin=log2(4)=2
            ###Response:    module mux_4to1(
                              input [3:0] <input_name>,
                              input [1:0] select,
                              output reg <output_name>
                            );
                            always @(*)
                              case(select)
                                2'd0: <output_name> = <input_name>[0];
                                2'd1: <output_name> = <input_name>[1];
                                2'd2: <output_name> = <input_name>[2];
                                2'd3: <output_name> = <input_name>[3];
                                default: <output_name> = 'b0;
                              endcase
                            endmodule
          }}

        - User: {user_input}
        - <mux_type> could be any number from 3 to 1024. You should be able to give accurate code with correct values for any given <mux_type>
        - Note: 1. Use same style and syntax as the samples. Calculations should be mathematically correct. Recheck mathematical calculations twice.
                2. Verilog code should be in same sytax as above examples.
                3. Take care of calculated <bit_size>, <select_pin> and <pin_0_range> values in code.
                4. Last <pin_n_range> should always be equal <bit_size>.
                5. Perform all calculation steps carefully.
                6. Avoid multiple responses.
                7. Accuracy and precision are your priority.
                8. Provide response only with verilog code and nothing else.
                9. Generate all possible output pins. (you should always be able generate 1000 pins if user asked you to)
        """
    ),
    "mux_with_pin_size": PromptTemplate(
        input_variables=["user_input"],
        template="""
        You are a Verilog Programmer and you are very good at Mathematics. Retrieve required data correctly from user input.
        Get inputs <mux_type>, <input_width>, <input_name> and <output_name> from user input and perform all the following steps:
            1. Calculate total number of pins starting from 0:
                - Total_number_of_pins = <mux_type>
                - last_pin = <Total_number_of_pins>-1
                - Do not consider 'default' pin in the <Total_number_of_pins>.

            2. Calculate range values for perticular pin use following formulas:
                - A simple trick is first calculate pin_0_range and addition_factor
                  - pin_0_range=[<input_width>:0]
                  - addition_factor = (<input_width>+1)
                - Now keep adding <addition_factor> to the previous range values on both side in the bracket.
                - Suppose, pin_<n-1>_range=[x:y]
                  The next pin will always be: 
                        pin_<n>_range=[(x+<addition_factor>):(y+<addition_factor>)]   # <n> is pin number
                - Add <addition_factor> correctly and take care of mathematical summation for each pin.
                  Note:
                        - Always perform complete calculation process for this step from pin_0_range to pin_<last_pin>_range.
                        - Solve manually using mathematical rules, do not use any coding language.
                        - Validation step: The y value of current pin should be the consecutive next value of x value of previous.
                                           The difference between x and y values of a particular pin should be equial to the <input_width>.
                                           The x value of last pin; x value of pin_<last_pin>_range should be equal to the <bit_size>.
                        - Add <addition_factor> carefully and take care of mathematical calculations for each pin.
                        - Always perform validation steps and fix you calculations.
                - See below examples for more references.
                - Always respond with calculated values and do not use keywords.

            3. Calculate the <bit_size> based on the given mux type and input width, using this formula:
                - a = <mux_type>*<input_width>
                  b = <a> + <mux_type>
                  bit_size = <b> - 1
                - Always perfrom upper steps in given order.
                - Always make sure the calculations are correct.
                - Validation Step: Suppose the last pin range is denoted as pin_n__range=[x:y];
                                   The bit_size should always be equal to 'x', max range of the last pin.
                - Always perform validation step.
                - See below examples for reference.
                - This is provided by user as; <input_name>[<input_width>:$]
                - Here "$" indicates incrementation.

            4. Calculate select_pin count:
                - select_pin=log2(<mux_type>)       # log_base_2
                - Perform log base 2 and choose roof value.
                - Example:
                    1. If input is 9-to-1 mux, select_pin=log2(9)=4
                    2. If input is 3-to-1 mux, select_pin=log2(3)=2
                    3. If input is 8-to-1 mux, select_pin=log2(8)=3
                - Note: Put the calculated <select_pin> value in select pin declaration and with each pin.
            5. Calculate ouput pin range:
                - output_range = pin_0_range

        - Samples codes:
              ###Instruction: create/implement/write 8-to-1 mux with output <output_name> and input <input_name>[8:$]
              - Note: 8-to-1 this could be in form 8:1 or 8 to 1 or 8to1, do not get confused.
              ###Calculation:
                    input_width = 8  # from user input
                    a = 8*8 = 64
                    b = a+8 = 64+8 = 72
                    bit_size= b-1 = 72-1 = 71
                    pin_0_range=[<input_width>:0]=[8:0]
                    addition_factor=(<input_width>+1)=(8+1)=9
                    pin_1_range=<pin_0_range>+[<addition_factor>]=[(9+<addition_factor>):(0+<addition_factor>)]=[(8+9):(0+9)]=[17:9]
                    pin_2_range=<pin_1_range>+[<addition_factor>]=[(19+<addition_factor>):(10+<addition_factor>)]=[(17+9):(9+9)]=[26:18]
                    pin_3_range=<pin_2_range>+[<addition_factor>]=[(29+<addition_factor>):(20+<addition_factor>)]=[(26+9):(18+9)]=[35:27]
                    pin_4_range=<pin_3_range>+[<addition_factor>]=[(39+<addition_factor>):(30+<addition_factor>)]=[(35+9):(27+9)]=[44:36]
                    pin_5_range=<pin_4_range>+[<addition_factor>]=[(49+<addition_factor>):(40+<addition_factor>)]=[(44+9):(36+9)]=[53:45]
                    pin_6_range=<pin_5_range>+[<addition_factor>]=[(59+<addition_factor>):(50+<addition_factor>)]=[(53+9):(45+9)]=[62:54]
                    pin_7_range=<pin_6_range>+[<addition_factor>]=[(69+<addition_factor>):(60+<addition_factor>)]=[(62+9):(54+9)]=[71:63]

                    output_range = pin_0_range

                    select_pin=log2(8)=3
              ###Response:
                    module mux_8to1(
                    input [<bit_size>:0] <input_name>,
                    input [<select_pin>-1:0] select,
                    output reg [<output_range>] <output_name>
                  );
                  always @(*)
                    case(select)
                      <select_pin>'d0: <output_name> = <input_name><pin_0_range>;
                      <select_pin>'d1: <output_name> = <input_name><pin_1_range>;
                      <select_pin>'d2: <output_name> = <input_name><pin_2_range>;
                      <select_pin>'d3: <output_name> = <input_name><pin_3_range>;
                      <select_pin>'d4: <output_name> = <input_name><pin_4_range>;
                      <select_pin>'d5: <output_name> = <input_name><pin_5_range>;
                      <select_pin>'d6: <output_name> = <input_name><pin_6_range>;
                      <select_pin>'d7: <output_name> = <input_name><pin_7_range>;
                      default: <output_name> = 'b0;
                    endcase
                  endmodule

        - Following are some examples:
            ###Instruction: create 16:1 mux with output output and input v_in[5:$]
            ###Calculation:
                  input_width = 5 # From user input
                  a = 16*5 = 80
                  b = a+16 = 80+16 = 96
                  bit_size= b-1 = 96-1 =95

                  pin_0_range=[<input_width>:0]=[5:0]
                  addition_factor=(<input_width>+1)=(5+1)=6
                  pin_1_range=<pin_0_range>+[<addition_factor>]=[5+<addition_factor>:(0+<addition_factor>)]=[(5+6):(0+6)]=[11:6]
                  pin_2_range=<pin_1_range>+[<addition_factor>]=[11+<addition_factor>:(6+<addition_factor>)]=[(11+6):(6+6)]=[17:12]
                  pin_3_range=<pin_2_range>+[<addition_factor>]=[17+<addition_factor>:(12+<addition_factor>)]=[(17+6):(12+6)]=[23:18]
                  pin_4_range=<pin_3_range>+[<addition_factor>]=[23+<addition_factor>:(18+<addition_factor>)]=[(23+6):(18+6)]=[29:24]
                  pin_5_range=<pin_4_range>+[<addition_factor>]=[29+<addition_factor>:(24+<addition_factor>)]=[(29+6):(24+6)]=[35:30]
                  pin_6_range=<pin_5_range>+[<addition_factor>]=[35+<addition_factor>:(30+<addition_factor>)]=[(35+6):(30+6)]=[41:36]
                  pin_7_range=<pin_6_range>+[<addition_factor>]=[41+<addition_factor>:(36+<addition_factor>)]=[(41+6):(36+6)]=[47:42]
                  pin_8_range=<pin_7_range>+[<addition_factor>]=[47+<addition_factor>:(42+<addition_factor>)]=[(47+6):(42+6)]=[53:48]
                  pin_9_range=<pin_8_range>+[<addition_factor>]=[53+<addition_factor>:(48+<addition_factor>)]=[(53+6):(48+6)]=[59:54]
                  pin_10_range=<pin_9_range>+[<addition_factor>]=[59+<addition_factor>:(54+<addition_factor>)]=[(59+6):(54+6)]=[65:60]
                  pin_11_range=<pin_10_range>+[<addition_factor>]=[65+<addition_factor>:(60+<addition_factor>)]=[65+6):(60+6)]=[71:66]
                  pin_12_range=<pin_11_range>+[<addition_factor>]=[71+<addition_factor>:(66+<addition_factor>)]=[(71+6):(66+6)]=[77:72]
                  pin_13_range=<pin_12_range>+[<addition_factor>]=[77+<addition_factor>:(72+<addition_factor>)]=[(77+6):(72+6)]]=[83:78]
                  pin_14_range=<pin_13_range>+[<addition_factor>]=[83+<addition_factor>:(78+<addition_factor>)]=[(83+6):(78+6)]=[89:84]
                  pin_15_range=<pin_14_range>+[<addition_factor>]=[89+<addition_factor>:(84+<addition_factor>)]=[(89+6):(84+6)]=[95:90]

                  output_range = pin_0_range

                  select_pin=log2(16)=4
            ###Response:
                  module mux_16to1(
                      input [95:0] v_in,
                      input [3:0] select,
                      output reg [<output_range>] output
                    );
                    always @(*)
                      case(select)
                        4'd0: output = v_in[5:0];
                        4'd1: output = v_in[11:6];
                        4'd2: output = v_in[17:12];
                        4'd3: output = v_in[23:18];
                        4'd4: output = v_in[29:24];
                        4'd5: output = v_in[35:30];
                        4'd6: output = v_in[41:36];
                        4'd7: output = v_in[47:42];
                        4'd8: output = v_in[53:48];
                        4'd9: output = v_in[59:54];
                        4'd10: output = v_in[65:60];
                        4'd11: output = v_in[71:66];
                        4'd12: output = v_in[77:72];
                        4'd13: output = v_in[83:78];
                        4'd14: output = v_in[89:84];
                        4'd15: output = v_in[95:90];
                        default: output = 'b0;
                      endcase
                    endmodule

            ###Instruction: create 7 to 1 mux with output mux_out and input data_in[7:$]
            ###Calculation:
                  1. Calculate total number of pins:
                    - Total number of pins = 7
                    - Last pin = 7 - 1 = 6

                  2. Calculate range values for each pin:
                    - Input width = 7
                    - Pin 0 range = [7:0]
                    - Addition factor = 7 + 1 = 8
                    - Pin 1 range = [7 + 8:0 + 8] = [15:8]
                    - Pin 2 range = [15 + 8:8 + 8] = [23:16]
                    - Pin 3 range = [23 + 8:16 + 8] = [31:24]
                    - Pin 4 range = [31 + 8:24 + 8] = [39:32]
                    - Pin 5 range = [39 + 8:32 + 8] = [47:40]
                    - Pin 6 range = [47 + 8:40 + 8] = [55:48]

                  3. Calculate bit size:
                    - a = 7 * 7 = 49
                    - b = a + 7 = 49 + 7 = 56
                    - bit size = b - 1 = 56 - 1 = 55

                  4. Calculate select pin count:
                    - select pin = log2(7) = 3

            ###Response:
                  module mux_7to1(
                      input [55:0] data_in,
                      input [2:0] select,
                      output reg [<output_range>] mux_out
                  );

                  always @(*)
                      case(select)
                          3'd0: mux_out = data_in[7:0];
                          3'd1: mux_out = data_in[15:8];
                          3'd2: mux_out = data_in[23:16];
                          3'd3: mux_out = data_in[31:24];
                          3'd4: mux_out = data_in[39:32];
                          3'd5: mux_out = data_in[47:40];
                          3'd6: mux_out = data_in[55:48];
                          default: mux_out = 'b0;
                      endcase
                  endmodule


        - User: {user_input}
        - <mux_type> could be any number from 3 to 1024. You should be able to give accurate code with correct values for any given <mux_type>
        - First perform all the calulations carefully then write the verilog code.
        - Always respond with calculated value and do not use keywords.
        - Always respond with the same syntax as examples.
        - Do not count the Default Pin in Total Number of Pins.
        - Always recheck all the calculations you perform. Fix them if not correct.
        - Accuracy and precision are our priority.
        - Provide response with verilog code.
        """
    ),
    "general_query": PromptTemplate(
        input_variables=["user_input"],
        template="You are a personal assistant. {user_input}"
    ),
}

# Initialize the conversation chain with memory
conversation = ConversationChain(llm=llm, memory=memory)


# Classify use case function
def classify_use_case(user_input):
    classification_prompt = PromptTemplate(
        input_variables=["user_input"],
        template="""Classify the following user input into classes: general_query, verilog_counter, flip_flop, only_mux, and mux_with_pin_size.
                    - general_query: only for general questions.
                    - verilog_counter: if user ask to implement a counter.
                    - flip_flop: user asking for implementing flip-flop or flop.
                    - only_mux: input formate would be like, mux type followed by input and output names. If input name is alone.
                    - mux_with_pin_size: input formate would be like, mux type followed by input and output names + input name followed by pin size value as [<value>:$]. If input name has a square bracket with value in it.
                    - Respond with the class name only. User input: {user_input}"""
    )
    chain = LLMChain(llm=llm, prompt=classification_prompt)
    return chain.run(user_input=user_input).strip().lower()


# def extract_last_verilog_code(text):
#     # Regular expression to match Verilog code blocks
#     verilog_code_pattern = r"```verilog(.*?)```"

#     # Find all matches for Verilog code blocks
#     matches = re.findall(verilog_code_pattern, text, re.DOTALL)

#     # Return the last match if available, otherwise return None
#     return f"```verilog{matches[-1]}```" if matches else None

def extract_last_verilog_code(text):
    # Regular expression to match the last Verilog code block
    verilog_code_pattern = r"(module\s[\s\S]*?endmodule)"

    # Find all matches for the pattern
    matches = re.findall(verilog_code_pattern, text, re.DOTALL)

    # Return the last match if available, otherwise return None
    return matches[-1] if matches else None


# Main conversation loop
current_use_case = None
while True:
    user_input = input("You: ")

    # Check for empty input
    if not user_input.strip():
        print("Please enter any query or type 'exit'/'quit' to end.")
        continue

    if user_input.lower() in ["exit", "quit"]:
        break

    # Classify the use case and update if needed
    new_use_case = classify_use_case(user_input)
    if new_use_case != current_use_case:
        # print(f"Switching from {current_use_case} to {new_use_case}")
        current_use_case = new_use_case

    # Use the appropriate prompt template
    prompt = prompts[current_use_case]
    # print("prompt ", prompt)
    chain = LLMChain(llm=llm, prompt=prompt)
    # print("Expected keys:", chain.input_keys)
    # print("Provided inputs:", user_input)

    response = chain.run(user_input=user_input)
    output = extract_last_verilog_code(response)
    print(f"response: {response}")
    print(f"output: {output}")
