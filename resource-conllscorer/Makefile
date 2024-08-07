################################################################################
##                                                                            ##
##  This file is part of ModelBlocks. Copyright 2009, ModelBlocks developers. ##
##                                                                            ##
##  ModelBlocks is free software: you can redistribute it and/or modify       ##
##  it under the terms of the GNU General Public License as published by      ##
##  the Free Software Foundation, either version 3 of the License, or         ##
##  (at your option) any later version.                                       ##
##                                                                            ##
##  ModelBlocks is distributed in the hope that it will be useful,            ##
##  but WITHOUT ANY WARRANTY; without even the implied warranty of            ##
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             ##
##  GNU General Public License for more details.                              ##
##                                                                            ##
##  You should have received a copy of the GNU General Public License         ##
##  along with ModelBlocks.  If not, see <http://www.gnu.org/licenses/>.      ##
##                                                                            ##
################################################################################

################################################################################
#
#  Includes
#
#  Include statements to this file should be preceded by the following includes
#
#  include $(dir $(CURDIR))resource-general/Makefile
#  include $(RESOURCE-TOKENIZER)/Makefile
#  include $(RESOURCE-LTREES)/Makefile
#  include $(RESOURCE-GCG)/Makefile
#  include $(RESOURCE-LVPCFG)/Makefile
#  include $(RESOURCE-LCPARSE)/Makefile
#  include $(RESOURCE-BNC)/Makefile
#  include $(RESOURCE-TREEBANK)/Makefile
#  include $(RESOURCE-GIGAWORD)/Makefile
#  include $(RESOURCE-KENLM)/Makefile
#  include $(RESOURCE-SRILM)/Makefile
#  include $(RESOURCE-INCRSEM)/Makefile
#  include $(RESOURCE-COREF-CONLL)/Makefile
################################################################################

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:
.DELETE_ON_ERROR:

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-conllscorer-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
CONLLSCORERDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(CONLLSCORERDIR))))
RESOURCE-NAME := The ConLL Reference Coreference Scorer
define RESOURCE-DESCR =

NAME: The CoNLL Reference Coreference Scorer
MB POINTER FILE: config/user-conllscorer-directory.txt
AVAILABILITY: https://github.com/conll/reference-coreference-scorers
DESCRIPTION: Scorers for coreference metrics used in the CoNLL-2011 and 2012 shared tasks
endef
CONLLSCORER-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-CONLLSCORER= $(error $(CONLLSCORER-ERROR))
endif
endif


%/user-conllscorer-directory.txt: | %
	echo '/fs/project/lin-dept/compling/reference-coreference-scorers-master' > $@
    
################################################################################
#
#  
#
################################################################################

.PRECIOUS: %corefeval
%corefeval: $$(DEP-CONLLSCORER) \
            $(CONLLSCORERDIR)/scorer.pl \
             genmodel/$$(subst -,.,$$(basename $$(basename $$*)))$$(subst _,.,$$(suffix $$*))key.conll \
	           $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$(basename $$*))$$(subst _,.,$$(suffix $$*))preds.conll 
	perl $(word 1, $^) all $(word 2, $^) $(word 3, $^) > $@

## TO SPEC
#.PRECIOUS: %syneval
#%syneval:  bin/evalb \
#           $(RESOURCE-GENERAL)/srcmodel/new.prm \
#           genmodel/$$(subst -,.,$$(basename $$(basename $$*)))$$(subst _,.,$$(suffix $$*))linetrees \
#           $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$(basename $$*))$$(subst _,.,$$(suffix $$*))linetrees
#	$< -p $(word 2,$^) $(word 3,$^) $(word 4,$^) > $@

