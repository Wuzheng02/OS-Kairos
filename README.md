<p align="center">
    <img src="docs/images/logo.png" width = "400"  alt="" align=center />
</p>
<p align="center">
  <a target="_blank">
    <img src="https://github.com/thunlp/OpenAttack/workflows/Test/badge.svg?branch=master" alt="Github Runner Covergae Status">
  </a>
  <a href="" target="_blank">
    <img src="https://readthedocs.org/projects/openattack/badge/?version=latest" alt="ReadTheDoc Status">
  </a>
  <a  href=""  target="_blank">
    <img src="https://img.shields.io/pypi/v/OpenAttack?label=pypi" alt="PyPI version">
  </a>
  <a  href=""  target="_blank">
    <img src="https://img.shields.io/github/v/release/thunlp/OpenAttack" alt="GitHub release (latest by date)">  
  </a>
  <a target="_blank">
    <img alt="GitHub" src="https://img.shields.io/github/license/thunlp/OpenAttack">
  </a>
   <a target="_blank">
    <img src="https://img.shields.io/badge/PRs-Welcome-red" alt="PRs are Welcome">
  </a>
<!-- <br><br>
  <a href="https://openattack.readthedocs.io/" target="_blank">Documentation</a> • <a href="#features--uses">Features & Uses</a> • <a href="#usage-examples">Usage Examples</a> • <a href="#attack-models">Attack Models</a> • <a href="#toolkit-design">Toolkit Design</a> 
<br> -->
</p>



<div align="center">
<img src="docs/images/pipeline.png" alt="Centered Image" style="width:1000px;"/>
</div>



### 1. Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Wuzheng02/OS-Kairos
   ```
2. Navigate into the project directory:
   ```bash
   cd OS-Kairos
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Dataset and checkpoint link: [Google Drive](https://drive.google.com/drive/folders/1LPmzxVPE7JAhRjaKjfe8qLA_d8M0RUgG?usp=drive_link)

### 2. Reproducing the Main Results of Static Experiments(Ignore 3)
1. Navigate to the test script directory:
   ```bash
   cd test_script
   ```
2. After modifying any test file in the `test_script` folder, update the `test_path` and `agent` paths, then run the script.
3. Before running, make sure to update the absolute paths of the `image_path` key in the JSON file located in `test_path`.

### 3. Generating Data or Testing the Model on Real Android Devices(Ignore 2)
#### 3.1 Modify the `config/config.yaml` as follows:
- `max_steps` determines the maximum steps for a particular instruction.
- `policy_llm` is the path to the base model used in `makedata_mode`.
- `sft_llm` is the path to the model with scoring ability for use in `test_mode`.
- `save_path` and `json_name` are where you store the execution traces and related information. Modify them as shown in the example.
- `asset_path` is the path to the instruction file. Please modify according to the example.
- `eval_nums` specifies the number of instructions to read in one run.
- `mode` can be set to either `makedata_mode` or `test_mode`. 
  - In `test_mode`, there are four sub-modes:
    - `single_step`: Essentially static testing (can be replaced by the scripts in the `test_script` folder).
    - `the_entire_trajectory`: Real dynamic testing using only the `policy_lm` model.
    - `gpt_test`: Uses the `sft_lm` model and simulates human behavior through GPT-4 for human-machine interaction testing.
    - `realworld_test`: Uses the `sft_lm` model and involves actual human interaction with the device.

#### 3.2 Modify the `env.py` file:
- Adjust the functions related to `adb` commands. The current code is a reference because we need two SSH hops to connect the `adb` command from the server hosting the base model to the actual device. 
- If your machine supports local deployment of the base model, you don't need an SSH connection. If you can connect directly to the server hosting the model, you won't need `sshpass`.

#### 3.3 Android Device Setup:
1. Connect an Android device to your computer via Developer Mode.
2. Ensure you can control the Android device using `adb` commands through the command line.

#### 3.4 (If unable to deploy the base model locally):
- You must configure the local machine, which is connected to the Android device, as the server, and the machine hosting the model will be the client for SSH communication.

#### 3.5 Run the project:
   ```bash
   python run.py
   ```

### 4. (Optional) If your instructions include Chinese, refer to the installation of Android Keyboard:
- Install [Android Keyboard](https://github.com/senzhk/ADBKeyBoard).
- Modify the `adb_input_text` function in `env.py` as instructed in the comments.

### 5. Demo
<div align="center">
<img src="docs/images/demo.png" alt="Centered Image" style="width:1000px;"/>
</div>

## Citation

Please cite our [paper]() if you use this toolkit:

```
@article{cheng2025OS,
  title={OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents},
  year={20245
}
```

## Contributors
We thank all the contributors to this project. And more contributions are very welcome.

![Contributors](https://contrib.rocks/image?repo=Wuzheng02/OS-Kairos)
