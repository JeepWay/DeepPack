# enter conda environment and run python script
@echo off
call D:\anaconda3\Scripts\activate.bat
call conda activate deeppack

@echo train 5x5
call python main.py --task unit_square --bin_w 5 --bin_h 5 --iterations 300000
call python main.py --task rectangular --bin_w 5 --bin_h 5 --iterations 350000
call python main.py --task square      --bin_w 5 --bin_h 5 --iterations 350000


@echo test 5x5
call python test.py --task unit_square --bin_w 5 --bin_h 5 --sequence_type type1 --max_sequence 25 --iterations 5000
call python test.py --task square      --bin_w 5 --bin_h 5 --sequence_type type1 --max_sequence 1  --iterations 5000
call python test.py --task square      --bin_w 5 --bin_h 5 --sequence_type type2 --max_sequence 13 --iterations 5000
call python test.py --task square      --bin_w 5 --bin_h 5 --sequence_type type3 --max_sequence 8  --iterations 5000
call python test.py --task rectangular --bin_w 5 --bin_h 5 --sequence_type type1 --max_sequence 7  --iterations 5000
call python test.py --task rectangular --bin_w 5 --bin_h 5 --sequence_type type2 --max_sequence 3  --iterations 5000
call python test.py --task rectangular --bin_w 5 --bin_h 5 --sequence_type type3 --max_sequence 4  --iterations 5000


@REM @echo train 4x4
@REM call python main.py --task unit_square --bin_w 4 --bin_h 4 --iterations 200000
@REM call python main.py --task rectangular --bin_w 4 --bin_h 4 --iterations 300000
@REM call python main.py --task square      --bin_w 4 --bin_h 4 --iterations 300000


@REM @echo test 4x4
@REM call python test.py --task unit_square --bin_w 4 --bin_h 4 --sequence_type type1 --max_sequence 16 --iterations 5000
@REM call python test.py --task square      --bin_w 4 --bin_h 4 --sequence_type random --max_sequence 16  --iterations 20000
@REM call python test.py --task square      --bin_w 4 --bin_h 4 --sequence_type type1 --max_sequence 1  --iterations 5000
@REM call python test.py --task square      --bin_w 4 --bin_h 4 --sequence_type type2 --max_sequence 8 --iterations 5000
@REM call python test.py --task square      --bin_w 4 --bin_h 4 --sequence_type type3 --max_sequence 7  --iterations 5000
@REM call python test.py --task rectangular --bin_w 4 --bin_h 4 --sequence_type random --max_sequence 16  --iterations 20000
@REM call python test.py --task rectangular --bin_w 4 --bin_h 4 --sequence_type type1 --max_sequence 4  --iterations 5000
@REM call python test.py --task rectangular --bin_w 4 --bin_h 4 --sequence_type type2 --max_sequence 4  --iterations 5000
@REM call python test.py --task rectangular --bin_w 4 --bin_h 4 --sequence_type type3 --max_sequence 3  --iterations 5000
