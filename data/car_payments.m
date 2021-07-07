function amount_left = car_payments(monthly_payment,price,apr,no_of_years,plot_bool)
  interest_per_month = apr/12;
  number_of_months = no_of_years*12;
  principle=price;
  P_vector=zeros(1,number_of_months);
  for i = 1:number_of_months
    principle=principle-monthly_payment;
    principle=(1+interest_per_month)*principle;
    P_vector(i)=principle;
  end
  amount_left=principle;
  if plot_bool
    plot([1:number_of_months]/12, P_vector)
    xlabel('time (years)')
    ylabel('principle amount left ($)')
  end
end
