function theta = initializeWeight(rows, columns)
  epsilon = 0.3;
  theta = rand(columns, rows+1)*2*epsilon - 2*epsilon;
endfunction
