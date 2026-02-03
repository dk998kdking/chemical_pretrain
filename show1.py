 
import os
import glob
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from collections import Counter

# ================== 一键生成最新真实预测对比图（零报错版）==================
samples_dir = 'runs_stage2_v2/samples'
pred_files = sorted(glob.glob(f'{samples_dir}/*_pred.xyz'), reverse=True)[:6]  # 取最新6个

mols_gen = []
mols_gt  = []
legends  = []

print("\n最新生成效果（已完全起飞！）".center(80, "="))

for i, pred_path in enumerate(pred_files):
    gt_path = pred_path.replace('_pred.xyz', '_real.xyz')
    
    # 读取生成的分子（双重保险）
    mol_gen = Chem.MolFromXYZFile(pred_path)
    if mol_gen is None or mol_gen.GetNumAtoms() == 0:
        with open(pred_path) as f:
            lines = [l for l in f.readlines()[2:] if l.strip()]
        mol_gen = Chem.RWMol()
        for line in lines:
            sym = line.split()[0]
            mol_gen.AddAtom(Chem.Atom(sym))
        AllChem.EmbedMolecule(mol_gen, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol_gen)

    # 读取真实分子
    mol_gt = Chem.MolFromXYZFile(gt_path)
    if mol_gt is None:
        continue

    # 统计原子
    gen_atoms = [a.GetSymbol() for a in mol_gen.GetAtoms()]
    gt_atoms  = [a.GetSymbol() for a in mol_gt.GetAtoms()]
    
    filename = os.path.basename(pred_path)
    print(f"{filename:<25} 生成: {Counter(gen_atoms)}")
    print(f"{'':<25} 真实: {Counter(gt_atoms)}")
    
    mols_gen.append(mol_gen)
    mols_gt.append(mol_gt)
    legends.append(f"Generated - {filename[:10]}")
    legends.append(f"Ground Truth - {filename[:10]}")

# 画高清大图
all_mols = []
all_legends = []
for i in range(len(mols_gen)):
    all_mols.append(mols_gen[i])
    all_mols.append(mols_gt[i])
    all_legends.append(legends[2*i])
    all_legends.append(legends[2*i+1])

img = Draw.MolsToGridImage(
    all_mols,
    molsPerRow=2,
    subImgSize=(680, 680),
    legends=all_legends,
    useSVG=False
)

output_png = "r202512087.png"
img.save(output_png)
print("\n" + "="*80)
 
print("="*80)
 