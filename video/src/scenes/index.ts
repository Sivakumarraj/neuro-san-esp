import React from 'react';
import {S01Hook, S02Agent, S03Team, S04Gap} from './intro';
import {S05World, S06Hops, S07Fitness} from './measure';
import {S08Evolve, S09Gate} from './search';
import {S10Esp, S11Predictor, S12GateNull} from './esp';
import {S13Holdout, S14Result, S15FinePrint} from './results';
import {S16V2, S17Names, S18Run, S19Limits, S20Recap} from './outro';

export const SCENES: Record<string, React.FC> = {
  s01_hook: S01Hook,
  s02_agent: S02Agent,
  s03_team: S03Team,
  s04_gap: S04Gap,
  s05_world: S05World,
  s06_hops: S06Hops,
  s07_fitness: S07Fitness,
  s08_evolve: S08Evolve,
  s09_gate: S09Gate,
  s10_esp: S10Esp,
  s11_predictor: S11Predictor,
  s12_gate_null: S12GateNull,
  s13_holdout: S13Holdout,
  s14_result: S14Result,
  s15_fineprint: S15FinePrint,
  s16_v2: S16V2,
  s17_names: S17Names,
  s18_run: S18Run,
  s19_limits: S19Limits,
  s20_recap: S20Recap,
};
