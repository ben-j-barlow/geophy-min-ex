using D3Trees

a, ai = action_info(planner, b0, tree_in_info=true);
out = gen(m, s, a, m.rng);
o = out[:o];
sp = out[:sp];
bp, ui = update_info(up, b, a, o);


#tree = ai[:tree];
#inbrowser(D3Tree(tree, init_expand=1), "safari")
