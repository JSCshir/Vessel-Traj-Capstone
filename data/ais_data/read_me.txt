obc_tanker_cargo.csv holds all the cargo from and tanker vessels from vessel type around the OBC channel.

golden_datasetv4.csv is all tanker and cargo vessels that are: travelling on correct side of the channel, no large course changes, SOG > 3 kts, Nat Status is 0 (underway using engine), Length, beam > 0 meters, No missing data.

mid_datav4.csv is all tanker and cargo vessels that have 5 or more AIS pings, pings must be within 30 minutes of each other. Includes golden dataset.

anom_datav2.csv is all tanker and cargo vessels that are anomalous. See ___ for more details. Not included in mid dataset.