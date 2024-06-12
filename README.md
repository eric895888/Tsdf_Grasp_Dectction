numpy注意要能使用np.int只能使用1.20以下或是1.23.5 (已在requirement.txt指定版本)  目前使用1.22.4	
而 scikit-image==0.19.0 (已在requirement.txt指定版本)
pip install opencv-contrib-python==4.6.0.66
安裝ros: 
sudo sh bash ros1_noetic_install.sh 就會自動安裝

安裝Anaconda
sudo sh Anaconda3-2024.02-1-Linux-x86_64.sh 推薦安裝在/home/你的使用者名稱/anaconda3 裡面
conda未啟動解決方式：https://developer.huawei.com/consumer/cn/blog/topic/03940616429410292 
或手動啟動source /home/eric/anaconda3/bin/activate

conda create --name torch python=3.8.10
sudo apt-get install python3-dev #不安裝的話編譯c++會失敗

(上述Anaconda或以下方式二選一)使用venv或virtualenv建立環境都可以
官網下載並安裝好python3.8.10
建立環境
venv方式：
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
cd.. 回到上一層目錄

virtualenv方式：

pip3 install virtualenv
virtualenv venv --python=pythonx.x.x 
終端機到資料夾路徑中activate


#(推薦)安裝所需要的套件
conda env create --file environment.yaml --name 自行定義名字
conda env create --file environment.yaml

#(不推薦)或是使用作者提供requirements下列指令,但因為open3d版本問題跟skylearn套件改名問題要自行處理
#pip install -r requirements.txt

pip install numpy==1.22.4
pip install pytorch-ignite==0.5.0.post2
pip install pykdtree
pip install rospkg
#目前open3d要是0.12.0!!!!!!!!!不然可能有未知錯誤像是產生core dump 或是 segmentation fault 這東西非常容易版本問題跟其他套件產成錯誤根據gdb追蹤可能是底層cuda調用c++那邊容易出問題
只靠requirements.txt也許torch少了些其他東西下方為安裝指定torch版本(當下是2.2.2)
參考來源:https://pytorch.org/get-started/locally/
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
其他版本torch參考來源:https://pytorch.org/get-started/previous-versions/ #不要用stable
#其他版本torch範例 pip install torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0

假如：
在python中查看自己對應的版本理論上torch應該是2.2.2 cuda應該是11.8也就是cu118
終端機裡打上python啟動環境或是vscode中新增一份.py檔案
import torch
print(torch.__version__)
顯示 2.2.2+cu118
print(torch.version.cuda)
顯示  11.8
print(torch.backends.cudnn.version())
顯示  8700
如果沒顯示出數字表示安裝不成功


安裝torch-scatter額外函式庫在下方對應的版本如下：
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.2+cu118.html

pip install -e . //安裝vgn

python scripts/convonet_setup.py build_ext --inplace 編譯src裡的vgn資料夾c++給python,編譯完後會自動把build/lib.linux-x86_64-3.8/src/內的資料夾複製進scripts中的資料夾中並覆蓋
因為這步驟在windows底下會出錯只能在linux中使用推測因為windows會調用Microsoft Visual Studio裡的編譯器來做會導致出錯


夾取程式碼使用方式：
請改成你們的外參./Grasp_detection_GIGA/scripts/calibration_params/2024_03_05改變裡面的相機內參intrinsic_matrix 畸變參數distortion_coefficients 相機外參T_cam2gripper內部的值
使用ArUco_detection.py 去偵測aruco位置的外參並改掉TM_grasp_withoutGUI.py5中的參數
T_cam_task_m = Transform(Rotation.from_quat([0.0091755 ,  0.9995211 ,  0.00176319 ,-0.02950025]), [ 0.16363484, -0.14483834 , 0.44753983])
執行ArUco_detection.py可以輸出夾爪位置

查看tsdf可視化:
在終端機輸入roscore
再開一個終端機輸入 roslaunch realsense2_camera rs_camera.launch align_depth:=true depth_width:=640 depth_height:=480 depth_fps:=30 color_width:=640 color_height:=480 color_fps:=30 filters:=pointcloud 
設定解析度及深度解析度為640x480並且顯示點雲
再開另外一個終端機輸入rosrun rviz rviz可以看到畫面
視窗左上角點file->open config選panda_grasp_TM.rviz

#d如果遇到Could not load the Qt platform plugin "xcb"
#參考這頁各種解決方式https://github.com/NVlabs/instant-ngp/discussions/300
#我是使用 os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH") 解決

進入scripts資料夾
cd scripts
在終端機打上指令產生資料 範例是產生積木資料--scence pile表示散堆 --object-set 後面表示urdf物件模型要載入哪一個資料夾 --num-grasps代表次數 --num-proc 代表幾個process在跑 ./data/pile/blocks 是儲存路徑 --sim-gui加上去會顯示可視化界面不加上這指令會跑比較快
python generate_data_parallel.py --scene pile --object-set blocks --num-grasps 4000000 --num-proc 10 --save-scene ./data/pile/blocks

python construct_dataset_parallel.py --num-proc 1 --single-view --add-noise dex data/pile/sundries data/dataset/sundries
模擬：
VGN
python sim_grasp_multiple.py --num-view 1 --object-set blocks --scene pile --num-rounds 100 --sideview --add-noise dex --force --best --model data/models/vgn_conv.pth --type vgn --result-path /path/to/result --sim-gui

GIGA
python sim_grasp_multiple.py --num-view 1 --object-set sundries --scene pile --num-rounds 100 --sideview --add-noise dex --force --best --model data/models/giga_pile.pt --type giga --result-path /path/to/result --sim-gui

    
python sim_grasp_multiple.py --num-view 1 --object-set M24_60mm_bolt --scene pile  --num-rounds 100 --sideview --add-noise dex --force --best --model data/models/best_TT_boltM24_giga_0508_giga_val_acc=0.9458.pt --type giga --result-path /path/to/result --sim-gui
# Tsdf_Graspdectction
